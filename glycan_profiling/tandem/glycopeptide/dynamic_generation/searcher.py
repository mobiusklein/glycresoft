import time
try:
    from Queue import Empty
except ImportError:
    from queue import Empty

from glycan_profiling.task import TaskExecutionSequence
from glycan_profiling.chromatogram_tree import Unmodified

from .search_space import (
    Parser,
    serialize_workload,
    deserialize_workload)

from ...workload import WorkloadManager
from ...spectrum_match import MultiScoreSpectrumSolutionSet

from ..scoring import LogIntensityScorer
from ..glycopeptide_matcher import GlycopeptideMatcher


class MultiScoreGlycopeptideMatcher(GlycopeptideMatcher):
    solution_set_type = MultiScoreSpectrumSolutionSet


def workload_grouping(chunks, max_scans_per_workload=500, starting_index=0):
    workload = []
    total_scans_in_workload = 0
    i = starting_index
    n = len(chunks)
    while total_scans_in_workload < max_scans_per_workload and i < n:
        chunk = chunks[i]
        workload.append(chunk)
        total_scans_in_workload += len(chunk)
        i += 1
    return workload, i


class SpectrumBatcher(TaskExecutionSequence):
    def __init__(self, groups, out_queue, max_scans_per_workload=250):
        self.groups = groups
        self.max_scans_per_workload = max_scans_per_workload
        self.out_queue = out_queue
        self.done_event = self._make_event()

    def generate(self):
        groups = self.groups
        max_scans_per_workload = self.max_scans_per_workload
        group_n = len(groups)
        group_i = 0
        while group_i < group_n:
            group_i_prev = group_i
            chunk, group_i = workload_grouping(groups, max_scans_per_workload, group_i)
            yield chunk, group_i_prev, group_n

    def run(self):
        for batch in self.generate():
            self.out_queue.put(batch)
        self.done_event.set()


class BatchMapper(TaskExecutionSequence):
    def __init__(self, predictive_searchers, in_queue, out_queue, in_done_event,
                 precursor_error_tolerance=5e-6, mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        self.predictive_searchers = predictive_searchers
        self.precursor_error_tolerance = precursor_error_tolerance
        self.mass_shifts = mass_shifts
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def execute_task(self, task):
        chunk, group_i_prev, group_n = task
        for label, predictive_search in self.predictive_searchers:
            task = StructureMapper(
                chunk, group_i_prev, group_n, predictive_search,
                precursor_error_tolerance=self.precursor_error_tolerance,
                mass_shifts=self.mass_shifts)
            task.label = label
            self.out_queue.put(task)

    def run(self):
        has_work = True
        while has_work:
            try:
                task = self.in_queue.get(True, 5)
                self.execute_task(task)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class StructureMapper(TaskExecutionSequence):
    def __init__(self, chunk, group_i, group_n, predictive_search, precursor_error_tolerance=5e-6,
                 mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        self.chunk = chunk
        self.group_i = group_i
        self.group_n = group_n
        self.predictive_search = predictive_search
        self.seen = set()
        self.mass_shifts = mass_shifts
        self.precursor_error_tolerance = precursor_error_tolerance

    def _log_cache(self):
        predictive_search = self.predictive_search
        hits = predictive_search.peptide_glycosylator._cache_hit
        misses = predictive_search.peptide_glycosylator._cache_miss
        total = hits + misses
        if total > 0:
            self.log("Cache Performance: %d / %d (%0.2f%%)" % (hits, total, hits / float(total) * 100.0))

    def _prepare_scan(self, scan):
        return scan.convert()

    def map_structures(self):
        counter = 0
        workload = WorkloadManager()
        predictive_search = self.predictive_search
        start = time.time()
        total_work = 0
        for i, group in enumerate(self.chunk):
            lo = float('inf')
            hi = 0
            temp = []
            for g in group:
                g = self._prepare_scan(g)
                if g.id in self.seen:
                    raise ValueError("Repeated Scan %r" % g.id)
                self.seen.add(g.id)
                counter += 1
                mass = g.precursor_information.neutral_mass
                temp.append(g)
                if lo > mass:
                    lo = mass
                if hi < mass:
                    hi = mass
            group = temp
            solutions = predictive_search.handle_scan_group(
                group, mass_shifts=self.mass_shifts, precursor_error_tolerance=self.precursor_error_tolerance)
            total_work += solutions.total_work_required()
            if i % 10 == 0 and i != 0:
                self.log('... Mapped Group %d (%0.2f%%) %0.3f-%0.3f with %d Items (%d Total)' % (
                    i + self.group_i, i * 100.0 / len(self.chunk), lo, hi,
                    solutions.total_work_required(), total_work))
            workload.update(solutions)
        end = time.time()
        self.log("Mapping Completed (%0.2f Sec)" % (end - start))
        self._log_cache()
        predictive_search.reset()
        return workload

    def add_decoy_glycans(self, workload):
        for hit_id, record in workload.hit_map.items():
            record = record.to_decoy_glycan()
            for scan in workload.hit_to_scan_map[hit_id]:
                hit_type = workload.scan_hit_type_map[scan.id, hit_id]
                workload.add_scan_hit(scan, record, hit_type)
        return workload

    def run(self):
        workload = self.map_structures()
        self.add_decoy_glycans(workload)
        return workload


class MapperExecutor(TaskExecutionSequence):
    def __init__(self, in_queue, out_queue, in_done_event):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def execute_task(self, mapper_task):
        workload = mapper_task()
        matcher_task = SpectrumMatcher(
            workload, mapper_task.group_i, mapper_task.group_n)
        return matcher_task

    def run(self):
        has_work = True
        while has_work:
            try:
                mapper_task = self.in_queue.get(True, 5)
                matcher_task = self.execute_task(mapper_task)
                self.out_queue.put(matcher_task)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class SerializingMapperExecutor(MapperExecutor):
    process_name = 'glycopeptide-db-map'

    def __init__(self, predictive_searchers, scan_loader, in_queue, out_queue,
                 in_done_event):
        super(SerializingMapperExecutor, self).__init__(
            in_queue, out_queue, in_done_event)

        self.predictive_searchers = predictive_searchers
        self.scan_loader = scan_loader

    def execute_task(self, mapper_task):
        label = mapper_task.predictive_search
        mapper_task.predictive_search = self.predictive_searchers[label]
        workload = mapper_task()
        workload.pack()
        workload = serialize_workload(workload)
        matcher_task = SpectrumMatcher(
            workload, mapper_task.group_i, mapper_task.group_n)
        return matcher_task

    def run(self):
        self.try_set_process_name()
        return super(SerializingMapperExecutor, self).run()


class SpectrumMatcher(TaskExecutionSequence):
    def __init__(self, workload, group_i, group_n, scorer_type=None,
                 ipc_manager=None, n_processes=6, evaluation_kwargs=None, **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        self.workload = workload
        self.group_i = group_i
        self.group_n = group_n

        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.ipc_manager = ipc_manager
        self.n_processes = n_processes

    def score_spectra(self):
        matcher = MultiScoreGlycopeptideMatcher(
            [], self.scorer_type, None, Parser,
            ipc_manager=self.ipc_manager, n_processes=self.n_processes)

        target_solutions = []
        self.log("... %0.2f%%" % (max((self.group_i - 1), 0) * 100.0 / self.group_n), self.workload)
        lo, hi = self.workload.mass_range()
        self.log("... Query Mass Range: %0.2f-%0.2f" % (lo, hi))

        batches = list(self.workload.batches())
        running_total_work = 0
        total_work = self.workload.total_work_required()
        self.workload.clear()
        for i, batch in enumerate(batches):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work + batch.batch_size, total_work,
                ((running_total_work + batch.batch_size) * 100.) / float(total_work)))
            running_total_work += batch.batch_size
            target_scan_solution_map = matcher._evaluate_hit_groups(
                batch, **self.evaluation_kwargs)
            temp = matcher._collect_scan_solutions(
                target_scan_solution_map, batch.scan_map)
            temp = [case for case in temp if len(case) > 0]
            for case in temp:
                case.simplify()
                case.select_top()
            target_solutions.extend(temp)
            batch.clear()
        return target_solutions

    def run(self):
        solution_sets = self.score_spectra()
        return solution_sets


class MatcherExecutor(TaskExecutionSequence):
    def __init__(self, in_queue, out_queue, in_done_event, scorer_type=None, ipc_manager=None,
                 n_processes=6, evaluation_kwargs=None, **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}

        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.n_processes = n_processes
        self.ipc_manager = ipc_manager

    def execute_task(self, matcher_task):
        matcher_task.ipc_manager = self.ipc_manager
        matcher_task.n_processes = self.n_processes
        matcher_task.scorer_type = self.scorer_type
        matcher_task.evaluation_kwargs = self.evaluation_kwargs
        solutions = matcher_task()
        return solutions

    def run(self):
        has_work = True
        while has_work:
            try:
                matcher_task = self.in_queue.get(True, 3)
                solutions = self.execute_task(matcher_task)
                self.out_queue.put(solutions)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class WorkloadUnpackingMatcherExecutor(MatcherExecutor):
    def __init__(self, scan_loader, in_queue, out_queue, in_done_event, scorer_type=None,
                 ipc_manager=None, n_processes=6, evaluation_kwargs=None, **kwargs):
        super(WorkloadUnpackingMatcherExecutor, self).__init__(
            in_queue, out_queue, in_done_event, scorer_type, ipc_manager,
            n_processes, evaluation_kwargs, **kwargs)
        self.scan_loader = scan_loader

    def execute_task(self, matcher_task):
        workload = matcher_task.workload
        matcher_task.workload = deserialize_workload(
            workload,
            self.scan_loader)
        return super(WorkloadUnpackingMatcherExecutor, self).execute_task(matcher_task)
