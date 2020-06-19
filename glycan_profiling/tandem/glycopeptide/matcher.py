from collections import OrderedDict

import threading
try:
    from Queue import Queue as ThreadQueue
except ImportError:
    from queue import Queue as ThreadQueue

try:
    import cPickle as pickle
except ImportError:
    import pickle

from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.structure import (
    CachingGlycopeptideParser,
    SequenceReversingCachingGlycopeptideParser)

from ..spectrum_evaluation import TandemClusterEvaluatorBase, DEFAULT_BATCH_SIZE
from ..process_dispatcher import SpectrumIdentificationWorkerBase
from ..oxonium_ions import gscore_scanner


class GlycopeptideIdentificationWorker(SpectrumIdentificationWorkerBase):
    process_name = 'glycopeptide-identification-worker'

    def __init__(self, input_queue, output_queue, producer_done_event, consumer_done_event,
                 scorer_type, evaluation_args, spectrum_map, mass_shift_map, log_handler,
                 parser_type, solution_packer, cache_seeds=None):
        if cache_seeds is None:
            cache_seeds = {}
        SpectrumIdentificationWorkerBase.__init__(
            self, input_queue, output_queue, producer_done_event, consumer_done_event,
            scorer_type, evaluation_args, spectrum_map, mass_shift_map,
            log_handler=log_handler, solution_packer=solution_packer)
        self.parser = parser_type()
        self.cache_seeds = cache_seeds

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher

    def before_task(self):
        if self.cache_seeds is None:
            return
        cache_seeds = self.cache_seeds
        if isinstance(cache_seeds, (str, bytes)):
            cache_seeds = pickle.loads(cache_seeds)

        oxonium_cache_seed = cache_seeds.get('oxonium_ion_cache')
        if oxonium_cache_seed is not None:
            oxonium_cache_seed = pickle.loads(oxonium_cache_seed)
            from glycan_profiling.structure.structure_loader import oxonium_ion_cache
            oxonium_ion_cache.update(oxonium_cache_seed)

        n_glycan_stub_cache_seed = cache_seeds.get('n_glycan_stub_cache')
        if n_glycan_stub_cache_seed is not None:
            n_glycan_stub_cache_seed = pickle.loads(n_glycan_stub_cache_seed)
            from glycan_profiling.structure.structure_loader import CachingStubGlycopeptideStrategy
            CachingStubGlycopeptideStrategy.update(n_glycan_stub_cache_seed)


class PeptideMassFilteringDatabaseSearchMixin(object):

    def find_precursor_candidates(self, scan, error_tolerance=1e-5, probing_range=0,
                                  mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        peptide_filter = None
        hits = []
        intact_mass = scan.precursor_information.extracted_neutral_mass
        if self.peptide_mass_filter:
            peptide_filter = self.peptide_mass_filter.build_peptide_filter(
                scan, self.peptide_mass_filter.product_error_tolerance, mass_shift=mass_shift)
        for i in range(probing_range + 1):
            query_mass = intact_mass - (i * self.neutron_offset) - mass_shift.mass
            unfiltered_matches = self.search_database_for_precursors(query_mass, error_tolerance)
            if self.peptide_mass_filter:
                hits.extend(map(self._mark_hit, [match for match in unfiltered_matches if peptide_filter(
                            match.peptide_mass - (i * self.neutron_offset))]))
            else:
                hits.extend(map(self._mark_hit, unfiltered_matches))
        return hits


class GlycopeptideMatcher(PeptideMassFilteringDatabaseSearchMixin, TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, parser_type=None,
                 n_processes=5, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None, batch_size=DEFAULT_BATCH_SIZE, peptide_mass_filter=None,
                 trust_precursor_fits=True, cache_seeds=None):
        if parser_type is None:
            parser_type = self._default_parser_type()
        super(GlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, batch_size=batch_size, trust_precursor_fits=trust_precursor_fits)
        self.peptide_mass_filter = peptide_mass_filter
        self.parser_type = parser_type
        self.parser = parser_type()
        self.cache_seeds = cache_seeds

    def _default_parser_type(self):
        return CachingGlycopeptideParser

    def reset_parser(self):
        self.parser = self.parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher

    def _transform_matched_collection(self, solution_set_collection):
        cache = {}
        for solution_set in solution_set_collection:
            for sm in solution_set:
                target = sm.target
                if target.id in cache:
                    sm.target = cache[target.id]
                else:
                    sm.target = cache[target.id] = self.parser(target)
        return solution_set_collection

    @property
    def _worker_specification(self):
        return GlycopeptideIdentificationWorker, {
            "parser_type": self.parser_type,
            "cache_seeds": self.cache_seeds
        }


class DecoyGlycopeptideMatcher(GlycopeptideMatcher):
    def _default_parser_type(self):
        return SequenceReversingCachingGlycopeptideParser


class TargetDecoyInterleavingGlycopeptideMatcher(PeptideMassFilteringDatabaseSearchMixin, TandemClusterEvaluatorBase):
    '''Searches a single database against all spectra, where targets are
    database matches, and decoys are the reverse of the individual target
    glycopeptides.

    A spectrum has a best target match and a best decoy match tracked
    separately.

    This means that targets and decoys share the same glycan composition and
    peptide backbone mass, and ergo share stub glycopeptides. This may not produce
    "random" enough decoy matches.
    '''
    def __init__(self, tandem_cluster, scorer_type, structure_database, minimum_oxonium_ratio=0.05,
                 n_processes=5, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None, batch_size=DEFAULT_BATCH_SIZE, peptide_mass_filter=None,
                 trust_precursor_fits=True, cache_seeds=None):
        super(TargetDecoyInterleavingGlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, batch_size=batch_size,
            trust_precursor_fits=trust_precursor_fits, cache_seeds=cache_seeds)
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.peptide_mass_filter = peptide_mass_filter
        self.target_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, peptide_mass_filter=peptide_mass_filter,
            trust_precursor_fits=trust_precursor_fits, cache_seeds=cache_seeds)
        self.decoy_evaluator = DecoyGlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, peptide_mass_filter=peptide_mass_filter,
            trust_precursor_fits=trust_precursor_fits, cache_seeds=cache_seeds)

    def filter_for_oxonium_ions(self, error_tolerance=1e-5):
        keep = []
        for scan in self.tandem_cluster:
            minimum_mass = 0
            if scan.acquisition_information:
                try:
                    scan_windows = scan.acquisition_information[0]
                    window = scan_windows[0]
                    minimum_mass = window.lower
                except IndexError:
                    pass
            try:
                ratio = gscore_scanner(
                    peak_list=scan.deconvoluted_peak_set, error_tolerance=error_tolerance,
                    minimum_mass=minimum_mass)
            except Exception:
                self.error("An error occurred while calculating the G-score for \"%s\"" % scan.id)
                ratio = 0
            scan.oxonium_score = ratio
            if ratio >= self.minimum_oxonium_ratio:
                keep.append(scan)
            else:
                self.debug("... Skipping scan %s with G-score %f" % (scan.id, ratio))
        self.tandem_cluster = keep

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        # Map scans to target database
        workload = self.target_evaluator._map_scans_to_hits(
            scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        target_solutions = []
        total_work = workload.total_work_required()
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work + batch.batch_size, total_work,
                ((running_total_work + batch.batch_size) * 100.) / float(total_work)))
            target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            running_total_work += batch.batch_size
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(target_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            target_solutions += temp

        # Reuse mapped hits from target database using the decoy evaluator
        # since this assumes that the decoys will be direct reversals of
        # target sequences. The decoy evaluator will handle the reversals.
        decoy_solutions = []
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work + batch.batch_size, total_work,
                ((running_total_work + batch.batch_size) * 100.) / float(total_work)))

            decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(decoy_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            decoy_solutions += temp
            running_total_work += batch.batch_size
        return target_solutions, decoy_solutions

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, *args, **kwargs):
        target_out = []
        decoy_out = []

        self.filter_for_oxonium_ions(**kwargs)
        target_out, decoy_out = self.score_bunch(
            self.tandem_cluster, precursor_error_tolerance,
            simplify=simplify, *args, **kwargs)
        if simplify:
            for case in target_out:
                case.simplify()
                case.select_top()
            for case in decoy_out:
                case.simplify()
                case.select_top()
        target_out = [x for x in target_out if len(x) > 0]
        decoy_out = [x for x in decoy_out if len(x) > 0]
        return target_out, decoy_out


class CompetativeTargetDecoyInterleavingGlycopeptideMatcher(TargetDecoyInterleavingGlycopeptideMatcher):
    '''A variation of :class:`TargetDecoyInterleavingGlycopeptideMatcher` where
    a spectrum can have only one match which is either a target or a decoy.
    '''
    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        target_solutions, decoy_solutions = super(
            CompetativeTargetDecoyInterleavingGlycopeptideMatcher, self).score_bunch(
            scans, precursor_error_tolerance, simplify, *args, **kwargs)
        target_solutions = OrderedDict([(s.scan.id, s) for s in target_solutions])
        decoy_solutions = OrderedDict([(s.scan.id, s) for s in target_solutions])

        remove_target = []
        for key in target_solutions:
            try:
                if target_solutions[key].score > decoy_solutions[key].score:
                    decoy_solutions.pop(key)
                else:
                    remove_target.append(key)
            except KeyError:
                pass
        for key in remove_target:
            target_solutions.pop(key)
        return list(target_solutions.values()), list(decoy_solutions.values())


class ComparisonGlycopeptideMatcher(TargetDecoyInterleavingGlycopeptideMatcher):
    '''A variation of :class:`TargetDecoyInterleavingGlycopeptideMatcher` where
    targets and decoys are drawn from separate hypotheses, and decoys are taken
    verbatim from their database without be reversed.
    '''
    def __init__(self, tandem_cluster, scorer_type, target_structure_database, decoy_structure_database,
                 minimum_oxonium_ratio=0.05, n_processes=5, ipc_manager=None,
                 probing_range_for_missing_precursors=3, mass_shifts=None,
                 batch_size=DEFAULT_BATCH_SIZE, peptide_mass_filter=None,
                 trust_precursor_fits=True, cache_seeds=None):
        super(ComparisonGlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, target_structure_database,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, batch_size=batch_size, peptide_mass_filter=peptide_mass_filter,
            trust_precursor_fits=trust_precursor_fits, cache_seeds=cache_seeds)

        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type

        self.target_structure_database = target_structure_database
        self.decoy_structure_database = decoy_structure_database

        self.minimum_oxonium_ratio = minimum_oxonium_ratio

        self.target_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.target_structure_database,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, peptide_mass_filter=peptide_mass_filter,
            trust_precursor_fits=trust_precursor_fits)
        self.decoy_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.decoy_structure_database,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, peptide_mass_filter=peptide_mass_filter,
            trust_precursor_fits=trust_precursor_fits)

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        # Map scans to target database
        self.log("... Querying Targets")
        waiting_task_results = ThreadQueue()

        def decoy_query_task():
            self.log("... Querying Decoys")
            workload = self.decoy_evaluator._map_scans_to_hits(
                scans, precursor_error_tolerance)
            waiting_task_results.put(workload)

        workload = self.target_evaluator._map_scans_to_hits(
            scans, precursor_error_tolerance)

        # Execute the potentially disk-heavy task in the background while
        # evaluating the target spectrum matches.
        decoy_query_thread = threading.Thread(target=decoy_query_task)
        decoy_query_thread.start()

        # Evaluate mapped target hits
        target_solutions = []
        workload_graph = workload.compute_workloads()
        total_work = workload.total_work_required(workload_graph)
        running_total_work = 0

        for i, batch in enumerate(workload.batches(self.batch_size)):
            running_total_work += batch.batch_size
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))

            target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(target_scan_solution_map, batch.scan_map)
            temp = [case for case in temp if len(case) > 0]
            if simplify:
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            target_solutions.extend(temp)

        self.debug("... Waiting For Decoy Mapping")
        decoy_query_thread.join()
        # workload = self.decoy_evaluator._map_scans_to_hits(
        #     scans, precursor_error_tolerance)
        workload = waiting_task_results.get()
        # Evaluate mapped target hits
        decoy_solutions = []
        workload_graph = workload.compute_workloads()
        total_work = workload.total_work_required(workload_graph)
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            running_total_work += batch.batch_size
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))

            decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce decoy solutions
            temp = self._collect_scan_solutions(decoy_scan_solution_map, batch.scan_map)
            temp = [case for case in temp if len(case) > 0]
            if simplify:
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            decoy_solutions.extend(temp)
        return target_solutions, decoy_solutions
