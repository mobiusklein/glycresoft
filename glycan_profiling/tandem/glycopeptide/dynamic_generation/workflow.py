import multiprocessing

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

from glycan_profiling import serialize

from glycan_profiling.task import TaskBase
from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.structure import ScanStub

from glycan_profiling.database import disk_backed_database, mass_collection
from glycan_profiling.structure.structure_loader import PeptideDatabaseRecord

from ...temp_store import TempFileManager
from ...spectrum_evaluation import group_by_precursor_mass
from ..scoring import LogIntensityScorer

from .journal import JournalFileWriter, JournalingConsumer

from .search_space import (
    PeptideGlycosylator,
    PredictiveGlycopeptideSearch,
    GlycanCombinationRecord,
    StructureClassification)

from .searcher import (
    Pipeline, SpectrumBatcher, SerializingMapperExecutor,
    BatchMapper, WorkloadUnpackingMatcherExecutor)


def _determine_database_contents(path, hypothesis_id=1):
    db = disk_backed_database.PeptideDiskBackedStructureDatabase(path, hypothesis_id=hypothesis_id)
    glycan_classes = db.query(
        serialize.GlycanCombination,
        serialize.GlycanClass.name).join(
        serialize.GlycanCombination.components).join(
        serialize.GlycanComposition.structure_classes).group_by(
        serialize.GlycanClass.name).all()
    glycan_classes = {
        pair[1] for pair in glycan_classes
    }
    n_glycan = serialize.GlycanTypes.n_glycan in glycan_classes
    o_glycan = (serialize.GlycanTypes.o_glycan in glycan_classes or
                serialize.GlycanTypes.gag_linker in glycan_classes)
    return {
        'n_glycan': n_glycan,
        'o_glycan': o_glycan
    }


def make_memory_database_proxy_resolver(path, n_glycan=True, o_glycan=False, hypothesis_id=1):
    proxy = mass_collection.MassCollectionProxy(
        PeptideDatabaseProxyLoader(path, n_glycan, o_glycan, hypothesis_id))
    return proxy


class PeptideDatabaseProxyLoader(object):
    def __init__(self, path, n_glycan=True, o_glycan=True, hypothesis_id=1):
        self.path = path
        self.n_glycan = n_glycan
        self.o_glycan = o_glycan
        self.hypothesis_id = hypothesis_id

    def __call__(self):
        db = disk_backed_database.PeptideDiskBackedStructureDatabase(
            self.path, hypothesis_id=self.hypothesis_id)
        peptides = map(
            PeptideDatabaseRecord.from_record,
            db)
        if self.n_glycan and self.o_glycan:
            peptides = [
                rec for rec in peptides
                if rec.has_glycosylation_sites()
            ]
        elif self.n_glycan:
            peptides = [
                rec for rec in peptides
                if rec.n_glycosylation_sites
            ]
        elif self.o_glycan:
            peptides = [
                rec for rec in peptides
                if bool(rec.o_glycosylation_sites) or bool(rec.gagylation_sites)
            ]
        mem_db = mass_collection.NeutralMassDatabase(peptides)
        mem_db.session = db.session
        mem_db.hypothesis_id = db.hypothesis_id
        return mem_db


class MultipartGlycopeptideIdentification(TaskBase):
    def __init__(self, target_peptide_db, decoy_peptide_db, scan_loader, precursor_error_tolerance=5e-6,
                 product_error_tolerance=2e-5, batch_size=1000, scorer_type=None, mass_shifts=None, n_processes=6,
                 evaluation_kwargs=None, ipc_manager=None, file_manager=None, **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        if file_manager is None:
            file_manager = TempFileManager()
        elif isinstance(file_manager, str):
            file_manager = TempFileManager(file_manager)
        if ipc_manager is None:
            ipc_manager = multiprocessing.Manager()
        if isinstance(target_peptide_db, str):
            target_peptide_db = disk_backed_database.PeptideDiskBackedStructureDatabase(
                target_peptide_db, cache_size=100)
            target_peptide_db = mass_collection.TransformingMassCollectionAdapter(
                target_peptide_db, PeptideDatabaseRecord.from_record)
            decoy_peptide_db = disk_backed_database.PeptideDiskBackedStructureDatabase(
                decoy_peptide_db, cache_size=100)
            decoy_peptide_db = mass_collection.TransformingMassCollectionAdapter(
                decoy_peptide_db, PeptideDatabaseRecord.from_record)

        self.scorer_type = scorer_type
        self.scan_loader = scan_loader

        self.target_peptide_db = target_peptide_db
        self.decoy_peptide_db = decoy_peptide_db

        self.precursor_error_tolerance = precursor_error_tolerance
        self.product_error_tolerance = product_error_tolerance
        self.batch_size = batch_size

        self.mass_shifts = mass_shifts

        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.n_processes = n_processes
        self.ipc_manager = ipc_manager

        self.file_manager = file_manager
        self.journal_path = self.file_manager.get('glycopeptide-match-journal')
        self.journal_file = JournalFileWriter(self.journal_path)

    def build_scan_groups(self):
        pinfos = self.scan_loader.precursor_information()
        stubs = [ScanStub(pinfo, self.scan_loader) for pinfo in pinfos]
        groups = group_by_precursor_mass(stubs, 1e-4)
        return groups

    def build_predictive_searchers(self):
        glycan_combinations = GlycanCombinationRecord.from_hypothesis(
            self.target_peptide_db.session, self.target_peptide_db.hypothesis_id)

        generator = PeptideGlycosylator(
            self.target_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.target_peptide_target_glycan)
        target_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            coarse_threshold=1)

        generator = PeptideGlycosylator(
            self.decoy_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.decoy_peptide_target_glycan)
        decoy_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            coarse_threshold=1)
        return target_predictive_search, decoy_predictive_search

    def run_pipeline(self, scan_groups):
        (target_predictive_search,
         decoy_predictive_search) = self.build_predictive_searchers()

        spectrum_batcher = SpectrumBatcher(
            scan_groups,
            Queue(10),
            max_scans_per_workload=self.batch_size)

        mapping_batcher = BatchMapper(
            # map labels to be loaded in the mapper executor to avoid repeatedly
            # serializing the databases.
            [
                ('target', 'target'),
                ('decoy', 'decoy')
            ],
            spectrum_batcher.out_queue,
            multiprocessing.Queue(1),
            spectrum_batcher.done_event,
            precursor_error_tolerance=self.precursor_error_tolerance,
            mass_shifts=self.mass_shifts)
        mapping_batcher.done_event = multiprocessing.Event()

        mapping_executor = SerializingMapperExecutor(
            dict([
                ('target', target_predictive_search),
                ('decoy', decoy_predictive_search)
            ]),
            self.scan_loader,
            mapping_batcher.out_queue,
            multiprocessing.Queue(2),
            mapping_batcher.done_event)
        mapping_executor.done_event = multiprocessing.Event()

        matching_executor = WorkloadUnpackingMatcherExecutor(
            self.scan_loader,
            mapping_executor.out_queue,
            # multiprocessing.Queue(10),
            Queue(50),
            mapping_executor.done_event,
            scorer_type=self.scorer_type,
            ipc_manager=self.ipc_manager,
            n_processes=self.n_processes,
            evaluation_kwargs=self.evaluation_kwargs,
            error_tolerance=self.product_error_tolerance,
        )
        # matching_executor.done_event = multiprocessing.Event()

        journal_consumer = JournalingConsumer(
            self.journal_file,
            matching_executor.out_queue,
            matching_executor.done_event)

        mapping_executor.start(process=True)
        # matching_executor.start(process=True)

        pipeline = Pipeline([
            spectrum_batcher,
            mapping_batcher,
            mapping_executor,
            matching_executor,
            journal_consumer,
        ])

        pipeline.start()
        pipeline.join()

    def run(self):
        scan_groups = self.build_scan_groups()
        self.run_pipeline(scan_groups)
