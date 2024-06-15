import time

from queue import Empty, Full, Queue
from typing import List, Optional, Tuple, Type, Union, Dict, Set, TYPE_CHECKING

from multiprocessing import Event, Manager

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase

from ms_deisotope.data_source import ProcessedScan, ProcessedRandomAccessScanSource
# from ms_deisotope.output import ProcessedMSFileLoader

from glycresoft.structure.scan import ScanStub
from glycresoft.tandem.glycopeptide.scoring.base import GlycopeptideSpectrumMatcherBase

from glycresoft.task import TaskExecutionSequence
from glycresoft.chromatogram_tree import Unmodified

from .search_space import (
    Parser,
    PredictiveGlycopeptideSearch)

from ...workload import WorkloadManager
from ...spectrum_match.solution_set import MultiScoreSpectrumSolutionSet

from ..scoring import LogIntensityScorer
from ..matcher import GlycopeptideMatcher

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from multiprocessing.synchronize import Event as MPEvent

Full: Exception

class MultiScoreGlycopeptideMatcher(GlycopeptideMatcher):
    solution_set_type = MultiScoreSpectrumSolutionSet


def IsTask(cls):
    return cls


def workload_grouping(
        chunks: List[List[ProcessedScan]],
        max_scans_per_workload: int=500,
        starting_index: int=0) -> Tuple[List[List[ProcessedScan]], int]:
    """
    Gather together precursor mass batches of tandem mass spectra into a unit
    batch, starting from index ``starting_index``.

    Parameters
    ----------
    chunks : List[List[:class:`~.ProcessedScan`]]
        The precursor mass batched spectra
    max_scans_per_workload : int
        The total number of scans to include in a unit batch. This is a soft
        upper bound, a unit batch may have more than this if the next precursor
        mass batch is too large, but no additional batches will be added after
        that.
    starting_index : int
        The offset into ``chunks`` to start collecting batches from

    Returns
    -------
    unit_batch : List[List[:class:`~.ProcessedScan`]]
        The batch of precursor mass batches
    ending_index : int
        The offset into ``chunks`` that the next round should
        start from.
    """
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
    """
    Break a big list of scans into precursor mass batched blocks of
    spectrum groups with an approximate maximum size. Feeds raw workloads
    into the pipeline.
    """
    groups: List
    out_queue: Queue
    max_scans_per_workload: int

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
            if self.error_occurred():
                break
            while not self.error_occurred():
                try:
                    self.out_queue.put(batch, True, 5)
                    break
                except Full:
                    pass
        self.done_event.set()


class BatchMapper(TaskExecutionSequence):
    """
    Wrap scan groups from :class:`SpectrumBatcher` in :class:`StructureMapper` tasks
    and ships them to an appropriate work queue (usually another process).

    .. note::
        The StructureMapper could be applied with or without a database bound to it,
        and for an IPC consumer the database should not be bound, only labeled.
    """

    search_groups: List[List[ProcessedScan]]
    precursor_error_tolerance: float
    mass_shifts: List[MassShiftBase]

    in_queue: Queue
    out_queue: Dict[str, Queue]
    in_done_event: 'MPEvent'
    done_event: 'MPEvent'

    def __init__(self, search_groups, in_queue, out_queue, in_done_event,
                 precursor_error_tolerance=5e-6, mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        if not isinstance(out_queue, dict):
            for label, _group in search_groups:
                out_queue = {label: out_queue}
        self.search_groups = search_groups
        self.precursor_error_tolerance = precursor_error_tolerance
        self.mass_shifts = mass_shifts
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def out_queue_for_label(self, label):
        return self.out_queue[label]

    def execute_task(self, task: Tuple[List[List[ProcessedScan]], int, int]):
        chunk, group_i_prev, group_n = task
        for label, search_group in self.search_groups:
            task = StructureMapper(
                chunk, group_i_prev, group_n, search_group,
                precursor_error_tolerance=self.precursor_error_tolerance,
                mass_shifts=self.mass_shifts)
            task.label = label
            # Introduces a thread safety issue?
            task.unbind_scans()
            while not self.error_occurred():
                try:
                    self.out_queue_for_label(label).put(task, True, 5)
                    break
                except Full:
                    pass

    def run(self):
        has_work = True
        while has_work and not self.error_occurred():
            try:
                task = self.in_queue.get(True, 5)
                self.execute_task(task)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


@IsTask
class StructureMapper(TaskExecutionSequence[WorkloadManager]):
    """
    Map spectra against the database using a precursor filtering search strategy,
    generating a task graph of spectrum-structure-mass_shift relationships, a
    :class:`~.WorkloadManager` instance.
    """

    chunk: List[List[ProcessedScan]]
    group_i: int
    group_n: int
    predictive_search: Union[PredictiveGlycopeptideSearch, str]
    precursor_error_tolerance: float
    mass_shifts: List[MassShiftBase]
    seen: Set[str]

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

    def bind_scans(self, source: ProcessedRandomAccessScanSource):
        for group in self.chunk:
            for scan in group:
                scan.bind(source)

    def unbind_scans(self):
        for group in self.chunk:
            for scan in group:
                scan.unbind()

    def get_scan_source(self):
        for group in self.chunk:
            for scan in group:
                return scan.source

    def _log_cache(self):
        if False:
            predictive_search = self.predictive_search
            hits = predictive_search.peptide_glycosylator._cache_hit
            misses = predictive_search.peptide_glycosylator._cache_miss
            total = hits + misses
            if total > 5000:
                self.log("... Cache Performance: %d / %d (%0.2f%%)" % (hits, total, hits / float(total) * 100.0))

    def _prepare_scan(self, scan: Union[ScanStub, ProcessedScan]) -> ProcessedScan:
        try:
            return scan.convert()
        except AttributeError:
            if isinstance(scan, ProcessedScan):
                return scan
            else:
                raise

    def map_structures(self) -> WorkloadManager:
        counter = 0
        workload = WorkloadManager()
        predictive_search = self.predictive_search
        start = time.time()
        total_work = 0
        lo_min = float('inf')
        hi_max = 0
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
            if i % 500 == 0 and i != 0:
                self.log('... Mapped Group %d (%0.2f%%) %0.3f-%0.3f with %d Items (%d Total)' % (
                    i + self.group_i, i * 100.0 / len(self.chunk), lo, hi,
                    solutions.total_work_required(), total_work))
            lo_min = min(lo_min, lo)
            hi_max = max(hi_max, hi)
            workload.update(solutions)
        end = time.time()
        if counter:
            self.debug("... Mapping Completed %0.3f-%0.3f (%0.2f sec.)" %
                       (lo_min, hi_max, end - start))
        self._log_cache()
        predictive_search.reset()
        return workload

    def add_decoy_glycans(self, workload: WorkloadManager) -> WorkloadManager:
        items = list(workload.hit_map.items())
        for hit_id, record in items:
            record = record.to_decoy_glycan()
            for scan in workload.hit_to_scan_map[hit_id]:
                hit_type = workload.scan_hit_type_map[scan.id, hit_id]
                workload.add_scan_hit(scan, record, hit_type)
        return workload

    def run(self) -> WorkloadManager:
        workload = self.map_structures()
        workload.pack()
        self.add_decoy_glycans(workload)
        self.predictive_search.construct_peptide_groups(workload)
        return workload


class MapperExecutor(TaskExecutionSequence):
    """This task executor consumes batches of precursor mass-grouped spectra,
    and produces batches of glycopeptides matched to spectra.

    Its task type is :class:`StructureMapper`

    """

    scan_loader: ProcessedRandomAccessScanSource
    predictive_searchers: Dict[str, PredictiveGlycopeptideSearch]

    in_queue: Queue
    out_queue: Queue
    in_done_event: Event
    done_event: Event

    def __init__(self, predictive_searchers, scan_loader, in_queue, out_queue, in_done_event):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()
        self.scan_loader = scan_loader
        self.predictive_searchers = predictive_searchers

    def execute_task(self, mapper_task: StructureMapper) -> 'SpectrumMatcher':
        self.scan_loader.reset()
        # In case this came from a labeled batch mapper and not
        # attached to an actual dynamic glycopeptide generator
        batch_label = None
        label = mapper_task.predictive_search
        if isinstance(label, str):
            batch_label = label
            mapper_task.predictive_search = self.predictive_searchers[label]

        mapper_task.bind_scans(self.scan_loader)
        workload = mapper_task()
        self.scan_loader.reset()
        matcher_task = SpectrumMatcher(
            workload,
            mapper_task.group_i,
            mapper_task.group_n,
            batch_label=batch_label)
        return matcher_task

    def run(self):
        has_work = True
        strikes = 0
        while has_work and not self.error_occurred():
            try:
                mapper_task = self.in_queue.get(True, 5)
                matcher_task = self.execute_task(mapper_task)
                while not self.error_occurred():
                    try:
                        self.out_queue.put(matcher_task, True, 5)
                        break
                    except Full:
                        pass
                # Detach the scans from the scan source again.
                mapper_task.unbind_scans()

            except Empty:
                strikes += 1
                if strikes % 50 == 0 and strikes:
                    self.log("... %d iterations without new batches on %s, done event: %s" % (
                        strikes, self, self.in_done_event.is_set()))
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class SemaphoreBoundMapperExecutor(MapperExecutor):
    def __init__(self, semaphore, predictive_searchers, scan_loader, in_queue, out_queue,
                 in_done_event, tracking_directory=None):
        super(SemaphoreBoundMapperExecutor, self).__init__(
            predictive_searchers, scan_loader,
            in_queue, out_queue, in_done_event)
        self.semaphore = semaphore

    def execute_task(self, mapper_task):
        with self.semaphore:
            result = super(SemaphoreBoundMapperExecutor, self).execute_task(mapper_task)
        return result


@IsTask
class SpectrumMatcher(TaskExecutionSequence[List[MultiScoreSpectrumSolutionSet]]):
    """Actually execute the spectrum matching specified in a :class:`~.WorkloadManager`.

    .. note::
        This task may spin up additional processes if :attr:`n_processes` is greater than
        1, but it must be ~4 or better usually to have an appreciable speedup compared to
        a executing the matching in serial. IPC communication is expensive, no matter what.
    """

    workload: WorkloadManager
    group_i: int
    group_n: int
    scorer_type: Type[GlycopeptideSpectrumMatcherBase]
    ipc_manager: 'SyncManager'
    batch_label: Optional[str]

    n_processes: int
    mass_shifts: List[MassShiftBase]
    evaluation_kwargs: Dict
    cache_seeds: Optional[Dict]

    def __init__(self, workload, group_i, group_n, scorer_type=None,
                 ipc_manager=None, n_processes=6, mass_shifts=None,
                 evaluation_kwargs=None, cache_seeds=None,
                 batch_label: Optional[str]=None,
                 **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        self.workload = workload
        self.group_i = group_i
        self.group_n = group_n
        self.batch_label = batch_label

        self.mass_shifts = mass_shifts
        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.ipc_manager = ipc_manager
        self.n_processes = n_processes
        self.cache_seeds = cache_seeds

    def score_spectra(self) -> List[MultiScoreSpectrumSolutionSet]:
        matcher = MultiScoreGlycopeptideMatcher(
            [], self.scorer_type, None, Parser,
            ipc_manager=self.ipc_manager,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            cache_seeds=self.cache_seeds)

        target_solutions = []
        lo, hi = self.workload.mass_range()
        batches = list(self.workload.batches(matcher.batch_size))
        if self.workload.total_size:
            label = ''
            if self.batch_label:
                label = self.batch_label.title() + ' '
            self.log(
                f"... {label}{max((self.group_i - 1), 0) * 100.0 / self.group_n:0.2f}% "
                f"({lo:0.2f}-{hi:0.2f}) {self.workload}")


        n_batches = len(batches)
        running_total_work = 0
        total_work = self.workload.total_work_required()
        self.workload.clear()
        for i, batch in enumerate(batches):
            if batch.batch_size == 0:
                batch.clear()
                continue
            if n_batches > 1:
                self.log("... Batch %d (%d/%d) %0.2f%%" % (
                    i + 1, running_total_work + batch.batch_size, total_work,
                    ((running_total_work + batch.batch_size) * 100.) / float(total_work + 1)))
            running_total_work += batch.batch_size
            target_scan_solution_map = matcher.evaluate_hit_groups(
                batch, **self.evaluation_kwargs)

            temp = matcher.collect_scan_solutions(
                target_scan_solution_map, batch.scan_map)
            temp = [case for case in temp if len(case) > 0]
            for case in temp:
                case.simplify()
                # Don't run the select top filters for consistency. They seemed to
                # influence reproducibility.
            target_solutions.extend(temp)
            batch.clear()

        if batches:
            label = ''
            if self.batch_label:
                label = self.batch_label.title() + ' '
            self.log(f"... Finished {label}{max(self.group_i - 1, 0) * 100.0 / self.group_n:0.2f}%"
                     f" ({lo:0.2f}-{hi:0.2f})")
        return target_solutions

    def run(self):
        solution_sets = self.score_spectra()
        return solution_sets


class MatcherExecutor(TaskExecutionSequence):
    """This task executor consumes mappings from glycopeptide to scan and runs spectrum
    matching, scoring each glycopeptide against their matched spectra. It produces  scored
    spectrum matches.

    This type complements :class:`MapperExecutor`

    Its task type is :class:`SpectrumMatcher`
    """

    in_queue: Queue
    out_queue: Queue
    in_done_event: 'MPEvent'
    done_event: 'MPEvent'
    ipc_manager: 'SyncManager'

    ipc_manager: Manager
    scorer_type: Type[GlycopeptideSpectrumMatcherBase]

    n_processes: int
    mass_shifts: List[MassShiftBase]
    evaluation_kwargs: Dict
    cache_seeds: Optional[Dict]

    def __init__(self, in_queue, out_queue, in_done_event, scorer_type=None, ipc_manager=None,
                 n_processes=6, mass_shifts=None, evaluation_kwargs=None, cache_seeds=None,
                 **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}

        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

        self.mass_shifts = mass_shifts
        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.n_processes = n_processes
        self.ipc_manager = ipc_manager
        self.cache_seeds = cache_seeds

    def configure_task(self, matcher_task: SpectrumMatcher):
        matcher_task.ipc_manager = self.ipc_manager
        matcher_task.n_processes = self.n_processes
        matcher_task.scorer_type = self.scorer_type
        matcher_task.evaluation_kwargs = self.evaluation_kwargs
        matcher_task.mass_shifts = self.mass_shifts
        matcher_task.mass_shift_map = {m.name: m for m in self.mass_shifts}
        return matcher_task

    def execute_task(self, matcher_task):
        matcher_task = self.configure_task(matcher_task)
        solutions = matcher_task()
        return solutions

    def run(self):
        has_work = True
        while has_work and not self.error_occurred():
            try:
                matcher_task: SpectrumMatcher = self.in_queue.get(True, 3)
                solutions = self.execute_task(matcher_task)
                while not self.error_occurred():
                    try:
                        self.out_queue.put(solutions, True, 5)
                        break
                    except Full:
                        pass
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()


class SemaphoreBoundMatcherExecutor(MatcherExecutor):
    def __init__(self, semaphore, in_queue, out_queue, in_done_event, scorer_type=None,
                 ipc_manager=None, n_processes=6, mass_shifts=None, evaluation_kwargs=None,
                 cache_seeds=None, **kwargs):
        super(SemaphoreBoundMatcherExecutor, self).__init__(
            in_queue, out_queue, in_done_event, scorer_type, ipc_manager,
            n_processes, mass_shifts, evaluation_kwargs, cache_seeds=cache_seeds, **kwargs)
        self.semaphore = semaphore

    def execute_task(self, matcher_task):
        with self.semaphore:
            result = super(SemaphoreBoundMatcherExecutor, self).execute_task(matcher_task)
        return result
