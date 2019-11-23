import os
import multiprocessing
from collections import OrderedDict

try:
    from Queue import Queue
except ImportError:
    from queue import Queue

from glycan_profiling import serialize

from glycan_profiling.task import TaskBase, Pipeline
from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.structure import ScanStub

from glycan_profiling.database import disk_backed_database, mass_collection
from glycan_profiling.structure.structure_loader import PeptideDatabaseRecord
from glycan_profiling.structure.scan import ScanInformationLoader

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from ...chromatogram_mapping import ChromatogramMSMSMapper
from ...temp_store import TempFileManager
from ...spectrum_evaluation import group_by_precursor_mass
from ...spectrum_match import SpectrumMatchClassification
from ..scoring import LogIntensityScorer

from .journal import (
    JournalFileWriter,
    JournalFileReader,
    JournalingConsumer,
    SolutionSetGrouper)

from .search_space import (
    PeptideGlycosylator,
    PredictiveGlycopeptideSearch,
    GlycanCombinationRecord,
    StructureClassification)

from .searcher import (
    SpectrumBatcher, SerializingMapperExecutor,
    BatchMapper, WorkloadUnpackingMatcherExecutor)

from .multipart_fdr import GlycopeptideFDREstimator


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


def make_memory_database_proxy_resolver(path, n_glycan=None, o_glycan=None, hypothesis_id=1):
    if n_glycan is None or o_glycan is None:
        config = _determine_database_contents(path, hypothesis_id)
        if o_glycan is None:
            o_glycan = config['o_glycan']
        if n_glycan is None:
            n_glycan = config['n_glycan']
    proxy = mass_collection.MassCollectionProxy(
        PeptideDatabaseProxyLoader(path, n_glycan, o_glycan, hypothesis_id))
    return proxy


def make_disk_backed_peptide_database(path, hypothesis_id=1, **kwargs):
    peptide_db = disk_backed_database.PeptideDiskBackedStructureDatabase(
        path, cache_size=100, hypothesis_id=hypothesis_id)
    peptide_db = mass_collection.TransformingMassCollectionAdapter(
        peptide_db, PeptideDatabaseRecord.from_record)
    return peptide_db


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
        mem_db = disk_backed_database.InMemoryPeptideStructureDatabase(peptides, db)
        return mem_db


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))


class MultipartGlycopeptideIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, target_peptide_db, decoy_peptide_db, scan_loader,
                 mass_shifts=None, n_processes=6,
                 evaluation_kwargs=None, ipc_manager=None, file_manager=None,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 glycan_score_threshold=1.0, peptide_masses_per_scan=100, **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        if mass_shifts is None or not mass_shifts:
            mass_shifts = [Unmodified]
        if file_manager is None:
            file_manager = TempFileManager()
        elif isinstance(file_manager, str):
            file_manager = TempFileManager(file_manager)
        if ipc_manager is None:
            ipc_manager = multiprocessing.Manager()
        if isinstance(target_peptide_db, str):
            target_peptide_db = self.build_default_disk_backed_db_wrapper(target_peptide_db)
        if isinstance(decoy_peptide_db, str):
            decoy_peptide_db = self.build_default_disk_backed_db_wrapper(decoy_peptide_db)

        self.tandem_scans = tandem_scans
        self.scorer_type = scorer_type
        self.scan_loader = scan_loader

        self.target_peptide_db = target_peptide_db
        self.decoy_peptide_db = decoy_peptide_db

        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits

        self.glycan_score_threshold = glycan_score_threshold
        self.peptide_masses_per_scan = peptide_masses_per_scan

        self.precursor_error_tolerance = 5e-6
        self.product_error_tolerance = 2e-5
        self.batch_size = 1000

        self.mass_shifts = mass_shifts

        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.n_processes = n_processes
        self.ipc_manager = ipc_manager

        self.file_manager = file_manager
        self.journal_path = self.file_manager.get('glycopeptide-match-journal')

    @classmethod
    def build_default_disk_backed_db_wrapper(cls, path, **kwargs):
        peptide_db = make_disk_backed_peptide_database(path, **kwargs)
        return peptide_db

    @classmethod
    def build_default_memory_backed_db_wrapper(cls, path, **kwargs):
        peptide_db = make_memory_database_proxy_resolver(path, **kwargs)
        return peptide_db

    def build_scan_groups(self):
        if self.tandem_scans is None:
            pinfos = self.scan_loader.precursor_information()
            self.tandem_scans = [ScanStub(pinfo, self.scan_loader) for pinfo in pinfos]
        groups = group_by_precursor_mass(self.tandem_scans, 1e-4)
        return groups

    def build_predictive_searchers(self):
        glycan_combinations = GlycanCombinationRecord.from_hypothesis(
            self.target_peptide_db.session, self.target_peptide_db.hypothesis_id)

        if self.glycan_score_threshold > 0:
            min_fragments = 2
        else:
            min_fragments = 0

        generator = PeptideGlycosylator(
            self.target_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.target_peptide_target_glycan)
        target_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold, min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan)

        generator = PeptideGlycosylator(
            self.decoy_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.decoy_peptide_target_glycan)
        decoy_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold, min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan)
        return target_predictive_search, decoy_predictive_search

    def run_identification_pipeline(self, scan_groups):
        (target_predictive_search,
         decoy_predictive_search) = self.build_predictive_searchers()

        # combined_searcher = CompoundGlycopeptideSearch(
        #     [target_predictive_search, decoy_predictive_search])

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
                # ('combined', 'combined')
            ],
            spectrum_batcher.out_queue,
            multiprocessing.Queue(5),
            spectrum_batcher.done_event,
            precursor_error_tolerance=self.precursor_error_tolerance,
            mass_shifts=self.mass_shifts)
        mapping_batcher.done_event = multiprocessing.Event()

        tracking_dir = None
        if debug_mode:
            tracking_dir = self.file_manager.get("mapping-log")
        mapping_executor = SerializingMapperExecutor(
            OrderedDict([
                ('target', target_predictive_search),
                ('decoy', decoy_predictive_search)
                # ('combined', combined_searcher)
            ]),
            self.scan_loader,
            mapping_batcher.out_queue,
            multiprocessing.Queue(5),
            mapping_batcher.done_event,
            tracking_directory=tracking_dir,
        )
        mapping_executor.done_event = multiprocessing.Event()

        # If we wished to, we could run multiple MatcherExecutors in
        # separate processes, and have them feed their results to the
        # final consumer via IPC queue, but that requires that the
        # matching process is bottlenecked by IdentificationProcessDispatcher's
        # consumer (currently running in the main thread) but this will
        # require finding a more efficient serialization method to then
        # pass even more traffic to the JournalingConsumer, or give each
        # MatcherExecutor its own journal.
        matching_executor = WorkloadUnpackingMatcherExecutor(
            self.scan_loader,
            mapping_executor.out_queue,
            Queue(50),
            mapping_executor.done_event,
            scorer_type=self.scorer_type,
            ipc_manager=self.ipc_manager,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            evaluation_kwargs=self.evaluation_kwargs,
            error_tolerance=self.product_error_tolerance,
        )

        journal_writer = JournalFileWriter(self.journal_path)
        journal_consumer = JournalingConsumer(
            journal_writer,
            matching_executor.out_queue,
            matching_executor.done_event)

        mapping_executor.start(process=True)

        pipeline = Pipeline([
            spectrum_batcher,
            mapping_batcher,
            mapping_executor,
            matching_executor,
            journal_consumer,
        ])

        pipeline.start()
        pipeline.join()
        journal_writer.close()
        return journal_writer.solution_counter

    def search(self, precursor_error_tolerance=1e-5, simplify=True, batch_size=500, **kwargs):
        self.evaluation_kwargs.update(kwargs)
        self.product_error_tolerance = self.evaluation_kwargs.pop('error_tolerance', 2e-5)
        self.precursor_error_tolerance = precursor_error_tolerance
        self.batch_size = batch_size

        self.log("Building Scan Groups...")
        scan_groups = self.build_scan_groups()
        self.log("{:d} Scans, {:d} Scan Groups".format(
            len(self.tandem_scans), len(scan_groups)))
        self.log("Running Identification Pipeline...")
        total_solutions_count = self.run_identification_pipeline(
            scan_groups)
        self.log("Loading Spectrum Matches From Journal...")
        reader = enumerate(JournalFileReader(
            self.journal_path, scan_loader=ScanInformationLoader(self.scan_loader), mass_shift_map={
            m.name: m for m in self.mass_shifts
        }))
        solutions = []
        last = 0.1
        should_log = False
        for i, sol in reader:
            if i * 1.0 / total_solutions_count > last:
                should_log = True
                last += 0.1
            elif i % 5000 == 0:
                should_log = True
            if should_log:
                self.log("... %d/%d Solutions Loaded (%0.2f%%)" % (
                    i, total_solutions_count, i * 100.0 / total_solutions_count))
                should_log = False
            solutions.append(sol)
        self.log("Partitioning Spectrum Matches...")
        groups = SolutionSetGrouper(solutions)
        return groups

    def estimate_fdr(self, glycopeptide_spectrum_match_groups, *args, **kwargs):
        keys = [SpectrumMatchClassification[i] for i in range(4)]
        g = glycopeptide_spectrum_match_groups
        self.log("Running Target Decoy Analysis with %d targets and %d/%d/%d decoys" % (
            len(g[keys[0]]), len(g[keys[1]]), len(g[keys[2]]), len(g[keys[3]])))
        estimator = GlycopeptideFDREstimator(glycopeptide_spectrum_match_groups)
        groups = estimator.start()
        return groups

    def map_to_chromatograms(self, chromatograms, tandem_identifications,
                             precursor_error_tolerance=1e-5, threshold_fn=lambda x: x.q_value < 0.05,
                             entity_chromatogram_type=GlycopeptideChromatogram):
        self.log("Mapping MS/MS Identifications onto Chromatograms")
        self.log("%d Chromatograms" % len(chromatograms))
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance,
            self.scan_loader.convert_scan_id_to_retention_time)
        self.log("Assigning Solutions")
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        self.log("Distributing Orphan Spectrum Matches")
        mapper.distribute_orphans()
        self.log("Selecting Most Representative Matches")
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms, mapper.orphans
