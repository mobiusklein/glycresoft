import os
import multiprocessing
import ctypes
import datetime
from collections import OrderedDict
try:
    from Queue import Queue
except ImportError:
    from queue import Queue

from glycan_profiling import serialize

from glycan_profiling.task import TaskBase, Pipeline, MultiEvent, TaskExecutionSequence
from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.structure import ScanStub

from glycan_profiling.database import disk_backed_database, mass_collection
from glycan_profiling.structure.structure_loader import PeptideDatabaseRecord
from glycan_profiling.structure.scan import ScanInformationLoader
from glycan_profiling.composition_distribution_model.site_model import (
    GlycoproteomePriorAnnotator)

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
    BatchMapper, WorkloadUnpackingMatcherExecutor,
    MapperExecutor, SemaphoreBoundMatcherExecutor)

from .multipart_fdr import GlycopeptideFDREstimator, GlycopeptideFDREstimationStrategy


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
    loader = PeptideDatabaseProxyLoader(path, n_glycan, o_glycan, hypothesis_id)
    proxy = mass_collection.MassCollectionProxy(loader, loader.session_resolver, hypothesis_id)
    return proxy


def make_disk_backed_peptide_database(path, hypothesis_id=1, **kwargs):
    peptide_db = disk_backed_database.PeptideDiskBackedStructureDatabase(
        path, cache_size=100, hypothesis_id=hypothesis_id)
    peptide_db = mass_collection.TransformingMassCollectionAdapter(
        peptide_db, PeptideDatabaseRecord.from_record)
    return peptide_db


class PeptideDatabaseProxyLoader(TaskBase):
    def __init__(self, path, n_glycan=True, o_glycan=True, hypothesis_id=1):
        self.path = path
        self.n_glycan = n_glycan
        self.o_glycan = o_glycan
        self.hypothesis_id = hypothesis_id
        self._source_database = None

    @property
    def source_database(self):
        if self._source_database is None:
            self._source_database = disk_backed_database.PeptideDiskBackedStructureDatabase(
                self.path, hypothesis_id=self.hypothesis_id)
        return self._source_database

    def session_resolver(self):
        return self.source_database.session

    def hypothesis(self):
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
        for r in db:
            rec = PeptideDatabaseRecord.from_record(r)
            if filter_level == 1 and rec.n_glycosylation_sites:
                peptides.append(rec)
            elif filter_level == 2 and rec.o_glycosylation_sites:
                peptides.append(rec)
            elif filter_level == 0 and rec.has_glycosylation_sites():
                peptides.append(rec)

        mem_db = disk_backed_database.InMemoryPeptideStructureDatabase(peptides, db)
        return mem_db


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))


class MultipartGlycopeptideIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, target_peptide_db, decoy_peptide_db, scan_loader,
                 mass_shifts=None, n_processes=6,
                 evaluation_kwargs=None, ipc_manager=None, file_manager=None,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 glycan_score_threshold=1.0, peptide_masses_per_scan=100,
                 fdr_estimation_strategy=None, glycosylation_site_models_path=None,
                 cache_seeds=None, n_mapping_workers=1, **kwargs):
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
            ipc_manager = multiprocessing.Manager()
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
        self.n_mapping_workers = n_mapping_workers
        self.ipc_manager = ipc_manager
        self.cache_seeds = cache_seeds

        self.file_manager = file_manager
        self.journal_path = self.file_manager.get('glycopeptide-match-journal')
        self.journal_path_collection = []
        self.glycosylation_site_models_path = glycosylation_site_models_path

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

        generator = PeptideGlycosylator(
            self.target_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.target_peptide_target_glycan,
            glycan_prior_model=glycan_prior_model)
        target_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold, min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan)

        generator = PeptideGlycosylator(
            self.decoy_peptide_db, glycan_combinations,
            default_structure_type=StructureClassification.decoy_peptide_target_glycan,
            glycan_prior_model=glycan_prior_model)
        decoy_predictive_search = PredictiveGlycopeptideSearch(
            generator, product_error_tolerance=self.product_error_tolerance,
            glycan_score_threshold=self.glycan_score_threshold, min_fragments=min_fragments,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            peptide_masses_per_scan=self.peptide_masses_per_scan)
        return target_predictive_search, decoy_predictive_search

    def run_branching_identification_pipeline(self, scan_groups):
        (target_predictive_search,
         decoy_predictive_search) = self.build_predictive_searchers()

        spectrum_batcher = SpectrumBatcher(
            scan_groups,
            Queue(10),
            max_scans_per_workload=self.batch_size)

        common_queue = multiprocessing.Queue(5)
        label_to_batch_queue = {
            "target": common_queue,
            "decoy": common_queue,
        }

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

        execution_branches = []
        approx_num_concurrent = max(int(self.n_processes / 6.0), 1)
        concurrent_branch_controller = multiprocessing.BoundedSemaphore(approx_num_concurrent)
        if self.n_mapping_workers == 1:
            n_workers_per_branch = self.n_processes
        else:
            n_workers_per_branch = self.n_processes / approx_num_concurrent
        self.log("... Creating %d Branches With %d Workers Per Branch" %
                 (self.n_mapping_workers * 2, n_workers_per_branch))
        for _i in range(self.n_mapping_workers * 2):
            journal_path_for = self.file_manager.get('glycopeptide-match-journal-%d' % (_i))
            self.log("... Branch %d Writing To %r" % (_i, journal_path_for))
            self.journal_path_collection.append(journal_path_for)
            done_event_for = multiprocessing.Event()
            branch = IdentificationWorkerBranch(
                common_queue,
                mapping_batcher.done_event,
                journal_path_for,
                concurrent_branch_controller,
                done_event_for,
                self.scan_loader, target_predictive_search, decoy_predictive_search,
                n_workers_per_branch,
                scorer_type=self.scorer_type,
                evaluation_kwargs=self.evaluation_kwargs,
                error_tolerance=self.product_error_tolerance,
                cache_seeds=self.cache_seeds,
                mass_shifts=self.mass_shifts)
            execution_branches.append(branch)

        pipeline = Pipeline([
            spectrum_batcher,
            mapping_batcher,
        ] + execution_branches)
        for branch in execution_branches:
            branch.start(process=True, daemon=False)
        pipeline.start(daemon=True)
        pipeline.join()
        total = 0
        for branch in execution_branches:
            total += branch.results_processed.value
        return total

    def run_identification_pipeline(self, scan_groups):
        (target_predictive_search,
         decoy_predictive_search) = self.build_predictive_searchers()

        spectrum_batcher = SpectrumBatcher(
            scan_groups,
            Queue(10),
            max_scans_per_workload=self.batch_size)


        # common_queue = multiprocessing.Queue(5)
        label_to_batch_queue = {
            "target": multiprocessing.Queue(5),
            "decoy": multiprocessing.Queue(5),
        }

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

        tracking_dir = None
        if debug_mode:
            tracking_dir = self.file_manager.get("mapping-log")

        mapping_executor_out_queue = multiprocessing.Queue(5)

        mapping_executors = []
        for _i in range(self.n_mapping_workers):
            target_mapping_executor = SerializingMapperExecutor(
                OrderedDict([
                    ('target', target_predictive_search),
                ]),
                self.scan_loader,
                label_to_batch_queue['target'],
                mapping_executor_out_queue,
                mapping_batcher.done_event,
                tracking_directory=tracking_dir,
            )
            target_mapping_executor.done_event = multiprocessing.Event()

            decoy_mapping_executor = SerializingMapperExecutor(
                OrderedDict([
                    ('decoy', decoy_predictive_search),
                ]),
                self.scan_loader,
                label_to_batch_queue['decoy'],
                mapping_executor_out_queue,
                mapping_batcher.done_event,
                tracking_directory=tracking_dir,
            )
            decoy_mapping_executor.done_event = multiprocessing.Event()
            mapping_executors.append(target_mapping_executor)
            mapping_executors.append(decoy_mapping_executor)

        mapping_executor_done_event = MultiEvent([
            mapping_executor.done_event for mapping_executor in mapping_executors
        ])

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
            mapping_executor_out_queue,
            Queue(5),
            mapping_executor_done_event,
            scorer_type=self.scorer_type,
            ipc_manager=self.ipc_manager,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            evaluation_kwargs=self.evaluation_kwargs,
            error_tolerance=self.product_error_tolerance,
            cache_seeds=self.cache_seeds
        )

        journal_writer = JournalFileWriter(self.journal_path)
        journal_consumer = JournalingConsumer(
            journal_writer,
            matching_executor.out_queue,
            matching_executor.done_event)

        # Launch the sequences that execute in separate processes
        # before launching the thread chain.
        for mapping_executor in mapping_executors:
            mapping_executor.start(process=True, daemon=True)

        pipeline = Pipeline([
            spectrum_batcher,
            mapping_batcher,
        ] + mapping_executors + [
            matching_executor,
            journal_consumer,
        ])
        self.journal_path_collection.append(self.journal_path)
        pipeline.start(daemon=True)
        pipeline.join()
        journal_writer.close()
        return journal_writer.solution_counter

    def _load_identifications_from_journal(self, journal_path, total_solutions_count, accumulator=None):
        if accumulator is None:
            accumulator = []
        reader = enumerate(JournalFileReader(
            journal_path,
            scan_loader=ScanInformationLoader(self.scan_loader),
            mass_shift_map={m.name: m for m in self.mass_shifts}), len(accumulator))
        last = 0.1
        should_log = False
        for i, sol in reader:
            if i * 1.0 / total_solutions_count > last:
                should_log = True
                last += 0.1
            elif i % 100000 == 0:
                should_log = True
            if should_log:
                self.log("... %d/%d Solutions Loaded (%0.2f%%)" % (
                    i, total_solutions_count, i * 100.0 / total_solutions_count))
                should_log = False
            accumulator.append(sol)
        return accumulator

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
        start_time = datetime.datetime.now()
        # total_solutions_count = self.run_identification_pipeline(scan_groups)
        total_solutions_count = self.run_branching_identification_pipeline(scan_groups)
        end_time = datetime.datetime.now()
        self.log("Database Search Complete, %s Elapsed" % (end_time - start_time))
        self.log("Loading Spectrum Matches From Journal...")
        solutions = []
        for journal_path in self.journal_path_collection:
            self._load_identifications_from_journal(journal_path, total_solutions_count, solutions)
        self.log("Partitioning Spectrum Matches...")
        groups = SolutionSetGrouper(solutions)
        return groups

    def estimate_fdr(self, glycopeptide_spectrum_match_groups, *args, **kwargs):
        keys = [SpectrumMatchClassification[i] for i in range(4)]
        g = glycopeptide_spectrum_match_groups
        self.log("Running Target Decoy Analysis with %d targets and %d/%d/%d decoys" % (
            len(g[keys[0]]), len(g[keys[1]]), len(g[keys[2]]), len(g[keys[3]])))
        estimator = GlycopeptideFDREstimator(
            glycopeptide_spectrum_match_groups, self.fdr_estimation_strategy)
        groups = estimator.start()
        return groups, estimator

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


class IdentificationWorkerBranch(TaskExecutionSequence):
    def __init__(self, input_batch_queue, input_done_event, journal_path, branch_semaphore, done_event,
                 # Mapping Executor Parameters
                 scan_loader=None, target_predictive_search=None, decoy_predictive_search=None,
                 # Matching Executor Parameters
                 n_processes=4, scorer_type=None, evaluation_kwargs=None, error_tolerance=None, cache_seeds=None,
                 mass_shifts=None, ):
        self.input_batch_queue = input_batch_queue
        self.input_done_event = input_done_event
        self.journal_path = journal_path
        self.branch_semaphore = branch_semaphore
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

    def run(self):
        ipc_manager = multiprocessing.Manager()
        self.try_set_process_name("glycresoft-identification")
        mapping_executor = MapperExecutor(
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
            self.branch_semaphore,
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

        journal_writer = JournalFileWriter(self.journal_path)
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
        journal_writer.close()
        self.results_processed.value = journal_writer.solution_counter
        ipc_manager.shutdown()
