'''Types for managing the evaluation of spectrum matches, to
describe matches in bulk for serialization.
'''
import time
from collections import defaultdict

from typing import List, Mapping, Tuple, Any, Type

from ms_deisotope.data_source import ProcessedScan

from glycresoft.task import TaskBase

from glycresoft.chromatogram_tree import MassShift

from ..spectrum_match import SpectrumMatch, MultiScoreSpectrumMatch, ScoreSet
from .task import TaskDeque


class SpectrumEvaluatorBase(object):
    """Abstract base class for abstracting away how spectrum matches are
    set up.

    Attributes
    ----------
    scan_map: Mapping
        A mapping from scan id to :class:`~.ProcessedScan`
    mass_shift_map: Mapping
        A mapping from mass shift name to :class:`~.MassShift`
    solution_map: Mapping
        A mapping from (scan id, mass shift name) to the packaged match.
    """
    scan_map: Mapping[str, ProcessedScan]
    mass_shift_map: Mapping[str, MassShift]
    solution_map: Mapping[Tuple[str, str], Any]

    def fetch_scan(self, key):
        return self.scan_map[key]

    def fetch_mass_shift(self, key):
        return self.mass_shift_map[key]

    def create_evaluation_context(self, group):
        return None

    def evaluate(self, scan: ProcessedScan, structure,
                 evaluation_context=None, *args, **kwargs):
        """Evaluate the match between ``structure`` and ``scan``

        Parameters
        ----------
        scan : ms_deisotope.ProcessedScan
            MSn scan to match against
        structure : object
            Structure to match against ``scan``
        evaluation_context: object, optional
            The evaluation context used to carry external information into the evaluation.
        *args
            Propagated to scoring function
        **kwargs
            Propagated to scoring function

        Returns
        -------
        SpectrumMatcherBase
        """
        raise NotImplementedError()

    def handle_instance(self, structure, scan, mass_shift, evaluation_context=None):
        solution = self.evaluate(scan, structure, mass_shift=mass_shift,
                                 evaluation_context=evaluation_context,
                                 **self.evaluation_args)
        self.solution_map[scan.id, mass_shift.name] = self.solution_packer(solution)
        return solution

    def construct_cache_subgroups(self, work_order):
        '''Build groups of structures which should be evaluated within the same context.
        The default implementation assumes each structure should get its own context.

        Subclasses should override this to determine how to properly group things to
        share context.

        Parameters
        ----------
        work_order: :class:`dict`
            A work order specification. Assumes the first entry in each order is the
            structure to be grouped.

        Returns
        -------
        subgroups: :class:`list` of :class:`list`
            Each distinct subgroup to be evauluated together.
        '''
        subgroups = []
        for key, order in work_order['work_orders'].items():
            record = order[0]
            subgroups.append([record])
        return subgroups

    def evaluate_subgroup(self, work_order, subgroup):
        """Evaluate a sub-group of structures with shared context against their respective
        spectra.

        Parameters
        ----------
        work_order : :class:`dict`
            The work order specification which includes the mapping from structure ID to
            spectrum hit type pairs.
        subgroup : :class:`list`
            The list of structures to evaluate together

        Returns
        -------
        results: :class:`list`
            A list of packed results for each member of the group
        """
        results = []
        evaluation_context = self.create_evaluation_context(subgroup)
        for structure in subgroup:
            scan_specification = work_order['work_orders'][structure.id][1]
            scan_specification = [
                (self.fetch_scan(i), self.fetch_mass_shift(j)) for i, j in scan_specification]
            solution_target = None
            solution = None
            for scan, mass_shift in scan_specification:
                solution = self.handle_instance(
                    structure, scan, mass_shift, evaluation_context)
                solution_target = solution.target
            if solution is not None:
                try:
                    solution.target.clear_caches()
                except AttributeError:
                    pass
            packed = self.pack_output(solution_target)
            results.append(packed)
        return results

    def handle_group(self, work_order):
        results = []
        hit_cache_groups = self.construct_cache_subgroups(work_order)
        for subgroup in hit_cache_groups:
            results.extend(self.evaluate_subgroup(work_order, subgroup))
        return results

    def handle_item(self, structure, scan_specification):
        scan_specification = [(self.fetch_scan(i), self.fetch_mass_shift(j)) for i, j in scan_specification]
        solution_target = None
        solution = None
        for scan, mass_shift in scan_specification:
            solution = self.handle_instance(structure, scan, mass_shift)
            solution_target = solution.target

        if solution is not None:
            try:
                solution.target.clear_caches()
            except AttributeError:
                pass
        return self.pack_output(solution_target)

    def pack_output(self, target):
        raise NotImplementedError()


class LocalSpectrumEvaluator(SpectrumEvaluatorBase, TaskBase):
    '''A :class:`SpectrumEvaluatorBase` implementation that counter-points
    :class:`IdentificationProcessDispatcher` by doing all of its evaluation within
    a single process.

    To avoid excessive subclassing, instances first try to steal the
    :meth:`construct_cache_subgroups` and :meth:`create_evaluation_context` from
    :attr:`evaluator`.
    '''

    log_process_cycle = 5000

    evaluator: SpectrumEvaluatorBase
    evaluation_args: dict


    def __init__(self, evaluator, scan_map, mass_shift_map, solution_packer, evaluation_args=None):
        if evaluation_args is None:
            evaluation_args = dict()
        self.evaluator = evaluator
        self.scan_map = scan_map
        self.mass_shift_map = mass_shift_map
        self.solution_packer = solution_packer
        self.solution_map = dict()
        self.evaluation_args = evaluation_args
        try:
            self.construct_cache_subgroups = self.evaluator.construct_cache_subgroups
        except AttributeError:
            self.log("Failed to acquire construct_cache_subgroups")
        try:
            self.create_evaluation_context = self.evaluator.create_evaluation_context
        except AttributeError:
            self.log("Failed to acquire create_evaluation_context")

    def evaluate(self, scan, structure, *args, **kwargs):
        return self.evaluator.evaluate(scan, structure, *args, **kwargs)

    def pack_output(self, target):
        package = (target.id, self.solution_map)
        self.solution_map = dict()
        return package

    def process(self, hit_map: Mapping[Any, Any],
                hit_to_scan_map: Mapping[Any, List[str]],
                scan_hit_type_map: Mapping[Tuple[str, Any], MassShift],
                hit_group_map: Mapping[int, List[Any]] = None) -> Mapping[str, List]:
        deque_builder = TaskDeque()
        deque_builder(hit_map, hit_to_scan_map,
                      scan_hit_type_map, hit_group_map)
        i = 0
        has_groups = bool(hit_group_map)
        if not has_groups:
            n = len(hit_to_scan_map)
            for target, scan_spec in deque_builder:
                i += 1
                if i % self.log_process_cycle == 0:
                    self.log("... %0.2f%% of Hits Searched (%d/%d)" %
                            (i * 100. / n, i, n))
                target_id, result = self.handle_item(target, scan_spec)
                target = hit_map[target_id]
                yield (target, result)
        else:
            n = len(hit_group_map)
            for work_order in deque_builder:
                i += 1
                if i % self.log_process_cycle == 0:
                    self.log("... %0.2f%% of Groups Searched (%d/%d)" %
                             (i * 100. / n, i, n))
                for target_id, result in self.handle_group(work_order):
                    target = hit_map[target_id]
                    yield (target, result)


class SequentialIdentificationProcessor(TaskBase):
    evaluation_method: SpectrumEvaluatorBase
    evaluation_args: dict

    mass_shift_map: Mapping[str, MassShift]
    structure_map: Mapping[Any, Any]
    scan_map: Mapping[str, ProcessedScan]

    solution_handler: 'SolutionHandler'
    solution_handler_type: Type['SolutionHandler']


    def __init__(self, evaluator, mass_shift_map, evaluation_args=None, solution_handler_type=None):
        if evaluation_args is None:
            evaluation_args = dict()
        if solution_handler_type is None:
            solution_handler_type = SolutionHandler
        self.evaluation_method = evaluator
        self.evaluation_args = evaluation_args
        self.mass_shift_map = mass_shift_map
        self.solution_handler_type = solution_handler_type
        self.solution_handler = self.solution_handler_type({}, {}, self.mass_shift_map)
        self.structure_map = None
        self.scan_map = None

    def _make_evaluator(self, **kwargs):
        evaluator = LocalSpectrumEvaluator(
            self.evaluation_method,
            self.scan_map,
            self.mass_shift_map,
            self.solution_handler.packer,
            self.evaluation_args)
        return evaluator

    def process(self, scan_map: Mapping[str, ProcessedScan],
                hit_map: Mapping[Any, Any],
                hit_to_scan_map: Mapping[Any, List[str]],
                scan_hit_type_map: Mapping[Tuple[str, Any], MassShift],
                hit_group_map: Mapping[int, List[Any]] = None) -> Mapping[str, List]:
        self.structure_map = hit_map
        start_time = time.time()
        self.scan_map = self.solution_handler.scan_map = scan_map
        evaluator = self._make_evaluator()
        # self.log("... Searching Hits (%d:%d)" % (
        #     len(hit_to_scan_map),
        #     sum(map(len, hit_to_scan_map.values()))))
        for target, score_map in evaluator.process(hit_map, hit_to_scan_map, scan_hit_type_map, hit_group_map):
            self.store_result(target, score_map)
        end_time = time.time()
        elapsed = end_time - start_time
        self.log("... Identification Completed (%0.2f sec.): %d Solutions" %
                 (elapsed, self.solution_handler.counter, ))
        return self.scan_solution_map

    @property
    def scan_solution_map(self):
        return self.solution_handler.scan_solution_map

    def store_result(self, target, score_map):
        """Save the spectrum match scores for ``target`` against the
        set of matched scans

        Parameters
        ----------
        target : object
            The structure that was matched
        score_map : dict
            Maps (scan id, mass shift name) to score
        """
        self.solution_handler(target, score_map)


class SolutionHandler(TaskBase):
    def __init__(self, scan_solution_map, scan_map, mass_shift_map, packer=None):
        if packer is None:
            packer = SolutionPacker()
        self.scan_solution_map = defaultdict(list, (scan_solution_map or {}))
        self.scan_map = scan_map
        self.mass_shift_map = mass_shift_map
        self.packer = packer
        self.counter = 0

    def _make_spectrum_match(self, scan_id, target, score, shift_type):
        return SpectrumMatch(
            self.scan_map[scan_id], target, score,
            mass_shift=self.mass_shift_map[shift_type])

    def store_result(self, target, score_map):
        """Save the spectrum match scores for ``target`` against the
        set of matched scans

        Parameters
        ----------
        target : object
            The structure that was matched
        score_map : dict
            Maps (scan id, mass shift name) to score
        """
        self.counter += 1
        for hit_spec, result_pack in score_map.items():
            scan_id, shift_type = hit_spec
            score = self.packer.unpack(result_pack)
            psm = self._make_spectrum_match(scan_id, target, score, shift_type)
            self.scan_solution_map[scan_id].append(psm)

    def __call__(self, target, score_map):
        return self.store_result(target, score_map)


class SolutionPacker(object):
    def __call__(self, spectrum_match):
        return spectrum_match.score

    def unpack(self, package):
        return package


class MultiScoreSolutionHandler(SolutionHandler):
    def __init__(self, scan_solution_map, scan_map, mass_shift_map, packer=None, spectrum_match_type=None):
        if spectrum_match_type is None:
            spectrum_match_type = MultiScoreSpectrumMatch
        if packer is None:
            packer = MultiScoreSolutionPacker(score_set_tp=spectrum_match_type.score_set_type)
        self.spectrum_match_type = spectrum_match_type
        super(MultiScoreSolutionHandler, self).__init__(
            scan_solution_map, scan_map, mass_shift_map, packer)

    def _make_spectrum_match(self, scan_id, target, score, shift_type):
        return self.spectrum_match_type(
            self.scan_map[scan_id], target, score,
            mass_shift=self.mass_shift_map[shift_type])


class MultiScoreSolutionPacker(object):
    def __init__(self, score_set_tp=None):
        if score_set_tp is None:
            score_set_tp = ScoreSet
        self.score_set_tp = score_set_tp

    def __call__(self, spectrum_match):
        return self.score_set_tp.from_spectrum_matcher(spectrum_match).pack()

    def unpack(self, package):
        return self.score_set_tp.unpack(package)
