import os
import multiprocessing
import threading
import ctypes
import datetime
import zlib
import pickle

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, OrderedDict
from multiprocessing.managers import SyncManager
from queue import Queue



from ms_deisotope.output.common import LCMSMSQueryInterfaceMixin
from ms_deisotope.data_source import ProcessedRandomAccessScanSource

from glycresoft import serialize
from glycresoft.serialize.hypothesis.hypothesis import GlycopeptideHypothesis


from glycresoft.tandem.spectrum_match.solution_set import MultiScoreSpectrumSolutionSet
from glycresoft.tandem.spectrum_match.spectrum_match import MultiScoreSpectrumMatch, SpectrumMatch

from glycresoft.task import (
    TaskBase, Pipeline,
    TaskExecutionSequence,
    make_shared_memory_manager,
    IPCLoggingManager, LoggingHandlerToken)

from glycresoft.chromatogram_tree import Unmodified, MassShift

from glycresoft.structure import ScanStub

from glycresoft.database import disk_backed_database, mass_collection
from glycresoft.structure.structure_loader import PeptideDatabaseRecord
from glycresoft.structure.scan import ScanInformationLoader
from glycresoft.composition_distribution_model.site_model import (
    GlycoproteomePriorAnnotator)

from glycresoft.tandem.oxonium_ions import OxoniumIndex, SignatureIonIndex, single_signatures, compound_signatures

from glycresoft.chromatogram_tree.chromatogram import Chromatogram, GlycopeptideChromatogram
from ...chromatogram_mapping import (
    ChromatogramMSMSMapper,
    TandemAnnotatedChromatogram,
    TandemSolutionsWithoutChromatogram
)
from ...temp_store import TempFileManager
from ...spectrum_evaluation import group_by_precursor_mass
from ...spectrum_match import SpectrumMatchClassification
from ...workflow import SearchEngineBase

from ..scoring import LogIntensityScorer, GlycopeptideSpectrumMatcherBase

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
    SpectrumBatcher,
    BatchMapper,
    SemaphoreBoundMatcherExecutor,
    SemaphoreBoundMapperExecutor)

from .multipart_fdr import GlycopeptideFDREstimator, GlycopeptideFDREstimationStrategy


def make_memory_database_proxy_resolver(path, n_glycan=None, o_glycan=None, hypothesis_id=1):
    if n_glycan is None or o_glycan is None:
        config = _determine_database_contents(path, hypothesis_id)
        if o_glycan is None:
            o_glycan = config['o_glycan']
        if n_glycan is None:
            n_glycan = config['n_glycan']
    loader = PeptideDatabaseProxyLoader(path, n_glycan, o_glycan, hypothesis_id)
    # Make it possible to load the session here without loading the peptides
    proxy = mass_collection.MassCollectionProxy(
        loader, loader.session_resolver, hypothesis_id, loader.hypothesis_resolver)
    return proxy


def make_disk_backed_peptide_database(path, hypothesis_id=1, **kwargs):
    peptide_db = disk_backed_database.PeptideDiskBackedStructureDatabase(
        path, cache_size=100, hypothesis_id=hypothesis_id)
    peptide_db = mass_collection.TransformingMassCollectionAdapter(
        peptide_db, PeptideDatabaseRecord.from_record)
    return peptide_db


class FetchManyIterator(object):
    def __init__(self, cursor, batch_size=1000000):
        self.cursor = cursor
        self.batch_size = batch_size

    def __iter__(self):
        result_set = self.cursor.fetchmany(self.batch_size)
        while result_set:
            for x in result_set:
                yield x
            result_set = self.cursor.fetchmany(self.batch_size)


class PeptideDatabaseProxyLoader(TaskBase):
    path: os.PathLike
    n_glycan: bool
    o_glycan: bool
    hypothesis_id: int

    def __init__(self, path, n_glycan=True, o_glycan=True, hypothesis_id=1):
        self.path = path
        self.n_glycan = n_glycan
        self.o_glycan = o_glycan
        self.hypothesis_id = hypothesis_id
        self._source_database = None

    @staticmethod
    def determine_database_glycan_types(path, hypothesis_id=1):
        return _determine_database_contents(path, hypothesis_id=hypothesis_id)

    @property
    def source_database(self):
        if self._source_database is None:
            self._source_database = disk_backed_database.PeptideDiskBackedStructureDatabase(
                self.path, hypothesis_id=self.hypothesis_id)
        return self._source_database

    def session_resolver(self):
        return self.source_database.session

    def hypothesis_resolver(self) -> GlycopeptideHypothesis:
        return self.source_database.hypothesis

    def __reduce__(self):
        return self.__class__, (self.path, self.n_glycan, self.o_glycan, self.hypothesis_id)

    def __call__(self):
        db = disk_backed_database.PeptideDiskBackedStructureDatabase(
            self.path, hypothesis_id=self.hypothesis_id)
        if self.n_glycan and self.o_glycan:
            filter_level = 0
        elif self.n_glycan:
            filter_level = 1
        elif self.o_glycan:
            filter_level = 2
        else:
            raise ValueError("Cannot determine how to filter peptides")
        peptides = []

        self.log("... Loading peptides from %r:%r" % (self.path, self.hypothesis_id))
        start = datetime.datetime.now()
        if filter_level == 1:
            has_sites = db.has_protein_sites()
            # This is an old database, have to do a full scan.
            if not has_sites:
                q = db.having_glycosylation_site()
                iterator = FetchManyIterator(db.session.execute(q))
            else:
                # Fast path for N-glycosylation sites which are marked, but where the bounds
                # aren't precise so the if statements are still needed.
                q = db.spanning_n_glycosylation_site()
                iterator = FetchManyIterator(db.session.execute(q))
        else:
            q = db.having_glycosylation_site()
            iterator = FetchManyIterator(db.session.execute(q))
        seen = set()
        for r in iterator:
            rec = PeptideDatabaseRecord.from_record(r)
            if rec.id in seen:
                self.log("Converted Peptide %d more than once!" % rec.id)
            seen.add(rec.id)
            if filter_level == 1 and rec.n_glycosylation_sites:
                peptides.append(rec)
            elif filter_level == 2 and rec.o_glycosylation_sites:
                peptides.append(rec)
            elif filter_level == 0 and rec.has_glycosylation_sites():
                peptides.append(rec)
        db.session.remove()
        PeptideDatabaseRecord.unshare_sites(peptides)
        end = datetime.datetime.now()
        elapsed = (end - start).total_seconds()
        self.log("... %0.2f seconds elapsed. Loaded %d peptides" % (elapsed, len(peptides)))
        mem_db = disk_backed_database.InMemoryPeptideStructureDatabase(peptides, db)
        return mem_db


def _determine_database_contents(path, hypothesis_id=1):
    db = disk_backed_database.PeptideDiskBackedStructureDatabase(
        path, hypothesis_id=hypothesis_id)
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


def make_peptide_glycosylator(path, hypothesis_id: int = 1, target_peptide: bool=True) -> PeptideGlycosylator:
    peptide_db = make_memory_database_proxy_resolver(path, hypothesis_id=hypothesis_id)
    gp_hypothesis: GlycopeptideHypothesis = peptide_db.session.query(
        GlycopeptideHypothesis).get(hypothesis_id)

    glycan_recs = GlycanCombinationRecord.from_hypothesis(
        peptide_db.session, gp_hypothesis.glycan_hypothesis_id)

    generator = PeptideGlycosylator(
        peptide_db,
        glycan_recs,
        default_structure_type=StructureClassification.target_peptide_target_glycan if target_peptide else
                               StructureClassification.decoy_peptide_target_glycan)
    return generator


def make_predictive_glycopeptide_search(path, hypothesis_id: int=1,
                                        target_peptide: bool=True,
                                        product_error_tolerance: float=2e-5,
                                        glycan_score_threshold: float=1.0,
                                        probing_range_for_missing_precursors: int=3,
                                        peptide_masses_per_scan: int=100,
                                        oxonium_ion_threshold: float=0.05) -> PredictiveGlycopeptideSearch:

    generator = make_peptide_glycosylator(
        path, hypothesis_id=hypothesis_id, target_peptide=target_peptide)

    if glycan_score_threshold > 0:
        min_fragments = 2
    else:
        min_fragments = 0

    predictive_search = PredictiveGlycopeptideSearch(
        generator,
        product_error_tolerance=product_error_tolerance,
        glycan_score_threshold=glycan_score_threshold,
        min_fragments=min_fragments,
        probing_range_for_missing_precursors=probing_range_for_missing_precursors,
        trust_precursor_fits=True,
        peptide_masses_per_scan=peptide_masses_per_scan,
        oxonium_ion_threshold=oxonium_ion_threshold)

    return predictive_search


class MultipartGlycopeptideIdentifier(SearchEngineBase):
    precursor_error_tolerance: float
    product_error_tolerance: float
    batch_size: int
    fdr_estimation_strategy: GlycopeptideFDREstimationStrategy
    scorer_type: Type[GlycopeptideSpectrumMatcherBase]
    evaluation_kwargs: Dict[str, Any]
    mass_shifts: List[MassShift]
    probing_range_for_missing_precursors: int
    trust_precursor_fits: bool
    glycan_score_threshold: float
    oxonium_threshold: float
    peptide_masses_per_scan: int

    ipc_manager: SyncManager
    file_manager: TempFileManager
    journal_path: str
    journal_path_collection: List[str]

    target_peptide_db: mass_collection.SearchableMassCollectionWrapper
    decoy_peptide_db: mass_collection.SearchableMassCollectionWrapper
    glycosylation_site_models_path: str

    cache_seeds: Any

    tandem_scans: List[ScanStub]
    scan_loader: Union[ProcessedRandomAccessScanSource,
                       LCMSMSQueryInterfaceMixin]

    n_processes: int

    def __init__(self,
                 tandem_scans,
                 scorer_type,
                 target_peptide_db,
                 decoy_peptide_db,
                 scan_loader,
                 mass_shifts=None,
                 n_processes=6,
                 evaluation_kwargs=None,
                 ipc_manager=None,
                 file_manager=None,
                 probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True,
                 glycan_score_threshold=1.0,
                 peptide_masses_per_scan=60,
                 fdr_estimation_strategy=None,
                 glycosylation_site_models_path=None,
                 cache_seeds=None,
                 oxonium_threshold: float=0.05,
                 **kwargs):
        if fdr_estimation_strategy is None:
            fdr_estimation_strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        if mass_shifts is None:
            mass_shifts = []
        if Unmodified not in mass_shifts:
            mass_shifts = [Unmodified] + list(mass_shifts)
        if file_manager is None:
            file_manager = TempFileManager()
        elif isinstance(file_manager, str):
            file_manager = TempFileManager(file_manager)
        if ipc_manager is None:
            ipc_manager = make_shared_memory_manager()
        if isinstance(target_peptide_db, str):
            target_peptide_db = self.build_default_disk_backed_db_wrapper(target_peptide_db)
        if isinstance(decoy_peptide_db, str):
            decoy_peptide_db = self.build_default_disk_backed_db_wrapper(decoy_peptide_db)

        self.tandem_scans = tandem_scans
        self.scan_loader = scan_loader

        self.scorer_type = scorer_type
        self.fdr_estimation_strategy = GlycopeptideFDREstimationStrategy[fdr_estimation_strategy]

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
        self.cache_seeds = cache_seeds

        self.file_manager = file_manager
        self.journal_path = self.file_manager.get('glycopeptide-match-journal')
        self.journal_path_collection = []
        self.glycosylation_site_models_path = glycosylation_site_models_path

        self.oxonium_threshold = oxonium_threshold

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

        glycan_prior_model = None
        if self.glycosylation_site_models_path is not None:
            self.log("Loading glycosylation site scoring models from %r" % self.glycosylation_site_models_path)
            glycan_prior_model = GlycoproteomePriorAnnotator.load(
                self.target_peptide_db.session,
                self.decoy_peptide_db.session,
                open(self.glycosylation_site_models_path, 'rt'))

        expand_combinatorics = True

        generator = PeptideGlycosylator(
            self.target_peptide_db,
            glycan_combinations,
            default_structure_type=StructureClassification.target_peptide_target_glycan,
            glycan_prior_model=glycan_prior_model,
            expand_combinatorics=expand_combinatorics
        )
        target_predictive_search = PredictiveGlycopeptideSearch(
            generator,
            product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold,
            min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan,
            oxonium_ion_threshold=self.oxonium_threshold)

        generator = PeptideGlycosylator(
            self.decoy_peptide_db,
            glycan_combinations,
            default_structure_type=StructureClassification.decoy_peptide_target_glycan,
            glycan_prior_model=glycan_prior_model,
            expand_combinatorics=expand_combinatorics
        )
        decoy_predictive_search = PredictiveGlycopeptideSearch(
            generator,
            product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold,
            min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan,
            oxonium_ion_threshold=self.oxonium_threshold)
        return target_predictive_search, decoy_predictive_search

    def run_branching_identification_pipeline(self, scan_groups):
        (target_predictive_search,
         decoy_predictive_search) = self.build_predictive_searchers()

        spectrum_batcher = SpectrumBatcher(
            scan_groups,
            Queue(10),
            max_scans_per_workload=self.batch_size)

        spectrum_batcher.start(daemon=True)

        common_queue = multiprocessing.Queue(5)
        label_to_batch_queue = {
            "target": common_queue,
            "decoy": common_queue,
        }

        ipc_logger = IPCLoggingManager()

        mapping_batcher = BatchMapper(
            # map labels to be loaded in the mapper executor to avoid repeatedly
            # serializing the databases.
            [
                ('target', 'target'),
                ('decoy', 'decoy')
            ],
            spectrum_batcher.out_queue,
            label_to_batch_queue,
            spectrum_batcher.done_event,
            precursor_error_tolerance=self.precursor_error_tolerance,
            mass_shifts=self.mass_shifts)
        mapping_batcher.done_event = multiprocessing.Event()
        mapping_batcher.start(daemon=True)

        execution_branches: List['IdentificationWorker'] = []
        scorer_type_payload = zlib.compress(pickle.dumps(self.scorer_type, -1), 9)
        predictive_search_payload = zlib.compress(
            pickle.dumps(
                (target_predictive_search, decoy_predictive_search), -1), 9)

        for i in range(self.n_processes):
            journal_path_for = self.file_manager.get('glycopeptide-match-journal-%d' % (i))
            self.journal_path_collection.append(journal_path_for)
            done_event_for = multiprocessing.Event()
            branch = IdentificationWorker(
                "IdentificationWorker-%d" % i,
                self.ipc_manager.address,
                common_queue,
                mapping_batcher.done_event,
                journal_path_for,
                done_event_for,
                self.scan_loader,
                target_predictive_search=predictive_search_payload,
                decoy_predictive_search=None,
                scorer_type=scorer_type_payload,
                evaluation_kwargs=self.evaluation_kwargs,
                error_tolerance=self.product_error_tolerance,
                cache_seeds=self.cache_seeds,
                mass_shifts=self.mass_shifts,
                log_handler=ipc_logger.sender(),
            )
            execution_branches.append(branch)
        # Clear these big blobs from the parent process, we no longer need them
        del scorer_type_payload
        del predictive_search_payload
        pipeline = Pipeline([
            spectrum_batcher,
            mapping_batcher,
        ] + execution_branches)

        for branch in execution_branches:
            try:
                branch.start(process=True, daemon=True)
            except Exception as err:
                self.log(f"... Error {err} occurred during worker startup")

        # At this point, all the components have started already, but
        # to let the Pipeline object setup its "started" invariants, call
        # `start` again, which is a no-op for already-started task sequences.
        pipeline.start(daemon=True)
        pipeline.join()
        ipc_logger.stop()
        had_error = pipeline.error_occurred()
        if had_error:
            message = "%d unrecoverable error%s occured during search!" % (
                had_error, 's' if had_error > 1 else '')
            pipeline.stop()
            common_queue.close()
            common_queue.cancel_join_thread()
            self.log("Terminating search pipeline")
            for branch in execution_branches:
                branch.kill_process()
            raise Exception(message)
        total = 0
        for branch in execution_branches:
            total += branch.results_processed.value
        return total

    def _load_identifications_from_journal(
            self, journal_path: os.PathLike,
            total_solutions_count: int,
            accumulator: Optional[List[MultiScoreSpectrumMatch]]=None) -> List[MultiScoreSpectrumMatch]:
        if accumulator is None:
            accumulator = []

        with self.scan_loader.toggle_peak_loading():
            reader = enumerate(JournalFileReader(
                journal_path,
                scan_loader=ScanInformationLoader(self.scan_loader),
                mass_shift_map={m.name: m for m in self.mass_shifts},
                score_set_type=self.scorer_type.get_score_set_type()),
                len(accumulator))
            i = float(len(accumulator))
            try:
                # Get the nearest progress checkpoint
                last = round(i / total_solutions_count, 1)
            except ZeroDivisionError:
                last = 0.1
            should_log = False
            for i, sol in reader:
                if i * 1.0 / total_solutions_count > last:
                    should_log = True
                    last += 0.1
                elif i % 100000 == 0 and i > 1:
                    should_log = True
                if should_log:
                    self.log("... %d/%d Solutions Loaded (%0.2f%%)" % (
                        i, total_solutions_count, i * 100.0 / total_solutions_count))
                    should_log = False
                accumulator.append(sol)
        return accumulator

    def search(self, precursor_error_tolerance: float=1e-5,
               simplify: bool=True,
               batch_size: int=500, **kwargs) -> SolutionSetGrouper:
        self.evaluation_kwargs.update(kwargs)
        self.product_error_tolerance = self.evaluation_kwargs.pop('error_tolerance', 2e-5)
        self.precursor_error_tolerance = precursor_error_tolerance
        self.batch_size = batch_size

        self.log("Building Scan Groups...")
        scan_groups = self.build_scan_groups()
        self.log("{:d} Scans, {:d} Scan Groups".format(
            len(self.tandem_scans), len(scan_groups)))
        self.log("Running Identification Pipeline...")
        start_time = datetime.datetime.now()
        total_solutions_count = self.run_branching_identification_pipeline(scan_groups)
        end_time = datetime.datetime.now()
        self.log("Database Search Complete, %s Elapsed" % (end_time - start_time))
        self.log("Loading Spectrum Matches From Journal...")
        solutions = []
        n = len(self.journal_path_collection)
        for i, journal_path in enumerate(self.journal_path_collection, 1):
            self.log("... Reading Journal Shard %s, %d/%d" % (journal_path, i, n))
            self._load_identifications_from_journal(journal_path, total_solutions_count, solutions)
        self.log("Partitioning Spectrum Matches...")
        groups = SolutionSetGrouper(solutions)
        return groups

    def estimate_fdr(self, glycopeptide_spectrum_match_groups: SolutionSetGrouper,
                     *args, **kwargs) -> Tuple[SolutionSetGrouper, GlycopeptideFDREstimator]:
        keys = [SpectrumMatchClassification[i] for i in range(4)]
        g = glycopeptide_spectrum_match_groups
        self.log("Running Target Decoy Analysis with %d targets and %d/%d/%d decoys" % (
            len(g[keys[0]]), len(g[keys[1]]), len(g[keys[2]]), len(g[keys[3]])))
        peptide_fdr_estimator = self.scorer_type.get_fdr_model_for_dimension('peptide')
        estimator = GlycopeptideFDREstimator(
            glycopeptide_spectrum_match_groups,
            self.fdr_estimation_strategy,
            peptide_fdr_estimator=peptide_fdr_estimator)
        groups: SolutionSetGrouper = estimator.start()
        self.log("Rebuilding Targets")
        cache = {}
        target_match_sets = groups.target_matches
        n = len(target_match_sets)
        for i, target_match_set in enumerate(target_match_sets):
            if i % 10000 == 0 and i:
                self.log("... Rebuilt %d Targets (%0.2f%%)" % (i, i * 100.0 / n))
            for target_match in target_match_set:
                if target_match.target.id in cache:
                    target_match.target = cache[target_match.target.id]
                else:
                    target_match.target = cache[target_match.target.id] = target_match.target.convert()
        cache.clear()
        return groups, estimator

    def map_to_chromatograms(
            self,
            chromatograms: List[Chromatogram],
            tandem_identifications: List[MultiScoreSpectrumSolutionSet],
            precursor_error_tolerance: float=1e-5,
            threshold_fn: Callable[[SpectrumMatch], bool]=lambda x: x.q_value < 0.05,
            entity_chromatogram_type: Type[GlycopeptideChromatogram]=GlycopeptideChromatogram
            ) -> Tuple[List[TandemAnnotatedChromatogram], List[TandemSolutionsWithoutChromatogram]]:
        self.log("Mapping MS/MS Identifications onto Chromatograms")
        self.log("%d Chromatograms" % len(chromatograms))
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance,
            self.scan_loader.convert_scan_id_to_retention_time)
        self.log("Assigning Solutions")
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        self.log("Distributing Orphan Spectrum Matches")
        mapper.distribute_orphans(threshold_fn=threshold_fn)
        self.log("Selecting Most Representative Matches")
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms, mapper.orphans


class IdentificationWorker(TaskExecutionSequence):
    name: str
    ipc_manager_address: Union[str, Tuple[str, int]]

    input_batch_queue: multiprocessing.Queue
    input_done_event: multiprocessing.Event

    done_event: multiprocessing.Event

    journal_path: os.PathLike
    scan_loader: Union[ProcessedRandomAccessScanSource,
                       LCMSMSQueryInterfaceMixin]

    target_predictive_search: PredictiveGlycopeptideSearch
    decoy_predictive_search: PredictiveGlycopeptideSearch

    n_processes: int
    scorer_type: Type[GlycopeptideSpectrumMatcherBase]

    scorer_type: Type[GlycopeptideSpectrumMatcherBase]
    evaluation_kwargs: Dict[str, Any]
    error_tolerance: float
    cache_seeds: Any
    mass_shifts: List[MassShift]

    log_handler: LoggingHandlerToken

    def __init__(self, name, ipc_manager_address, input_batch_queue, input_done_event,
                 journal_path, done_event,
                 # Mapping Executor Parameters
                 scan_loader=None, target_predictive_search=None, decoy_predictive_search=None,
                 # Matching Executor Parameters
                 n_processes=1, scorer_type=None, evaluation_kwargs=None, error_tolerance=None,
                 cache_seeds=None, mass_shifts=None,
                 log_handler=None):
        self.name = name
        self.ipc_manager_address = ipc_manager_address
        self.input_batch_queue = input_batch_queue
        self.input_done_event = input_done_event
        self.journal_path = journal_path
        self.done_event = done_event

        # Mapping Executor
        self.scan_loader = scan_loader
        self.target_predictive_search = target_predictive_search
        self.decoy_predictive_search = decoy_predictive_search

        # Matching Executor
        self.n_processes = n_processes
        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.error_tolerance = error_tolerance
        self.cache_seeds = cache_seeds
        self.mass_shifts = mass_shifts
        self.results_processed = multiprocessing.Value(ctypes.c_uint64)
        self.log_handler = log_handler

    def _get_repr_details(self):
        props = ["name=%r" % self.name, "pid=%r" % multiprocessing.current_process().pid]
        return ', '.join(props)

    def _name_for_execution_sequence(self):
        return self.name

    def run(self):
        self.try_set_process_name("glycresoft-identification")
        self.log_handler.add_handler()
        ipc_manager = SyncManager(self.ipc_manager_address)
        ipc_manager.connect()
        lock = threading.RLock()
        # Late loading of the compressed serialized scorer type to avoid balooning
        # the parent process
        if isinstance(self.scorer_type, (str, bytes)):
            self.scorer_type = pickle.loads(zlib.decompress(self.scorer_type))
        if isinstance(self.target_predictive_search, (str, bytes)):
            self.target_predictive_search, self.decoy_predictive_search = pickle.loads(
                zlib.decompress(self.target_predictive_search))

        oxonium_ion_index = OxoniumIndex()
        oxonium_ion_index.build_index(
            self.target_predictive_search.peptide_glycosylator.glycan_combinations, oxonium=True)

        signatures = list(single_signatures)
        if self.evaluation_kwargs.get("rare_signatures"):
            signatures.extend(compound_signatures)
        signature_index = SignatureIonIndex(signatures)
        signature_index.build_index(
            self.target_predictive_search.peptide_glycosylator.glycan_combinations)

        self.target_predictive_search.oxonium_ion_index = oxonium_ion_index
        self.target_predictive_search.signature_ion_index = signature_index
        self.decoy_predictive_search.oxonium_ion_index = oxonium_ion_index
        self.decoy_predictive_search.signature_ion_index = signature_index

        mapping_executor = SemaphoreBoundMapperExecutor(
            lock,
            OrderedDict([
                ('target', self.target_predictive_search),
                ('decoy', self.decoy_predictive_search)
            ]),
            self.scan_loader,
            self.input_batch_queue,
            Queue(5),
            self.input_done_event,
        )
        matching_executor = SemaphoreBoundMatcherExecutor(
            lock,
            mapping_executor.out_queue,
            Queue(5),
            mapping_executor.done_event,
            scorer_type=self.scorer_type,
            ipc_manager=ipc_manager,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            evaluation_kwargs=self.evaluation_kwargs,
            error_tolerance=self.error_tolerance,
            cache_seeds=self.cache_seeds
        )

        journal_writer = JournalFileWriter(
            self.journal_path, score_columns=self.scorer_type.get_score_set_type().field_names())
        journal_consumer = JournalingConsumer(
            journal_writer,
            matching_executor.out_queue,
            matching_executor.done_event)
        journal_consumer.done_event = self.done_event

        pipeline = Pipeline([
            mapping_executor,
            matching_executor,
            journal_consumer,
        ])
        pipeline.start(daemon=True)
        pipeline.join()
        if pipeline.error_occurred():
            self.log("An error occurred while executing %s" % (self, ))
            self.set_error_occurred()
        journal_writer.close()
        self.results_processed.value = journal_writer.solution_counter
        self.log("%s has finished. %d solutions calculated." %
                 (self, journal_writer.solution_counter, ))
