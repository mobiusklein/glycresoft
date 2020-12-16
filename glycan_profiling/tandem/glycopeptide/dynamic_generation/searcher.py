import os
import time
try:
    from Queue import Empty, Full
except ImportError:
    from queue import Empty, Full

from ms_deisotope.data_source import ProcessedScan

from glycan_profiling.task import TaskExecutionSequence
from glycan_profiling.chromatogram_tree import Unmodified

from .search_space import (
    Parser,
    serialize_workload,
    deserialize_workload)

from ...workload import WorkloadManager
from ...spectrum_match.solution_set import MultiScoreSpectrumSolutionSet

from ..scoring import LogIntensityScorer
from ..matcher import GlycopeptideMatcher


class MultiScoreGlycopeptideMatcher(GlycopeptideMatcher):
    solution_set_type = MultiScoreSpectrumSolutionSet


def IsTask(cls):
    return cls


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


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))
memory_debug = bool(os.environ.get("GLYCRESOFTDEBUGMEMORY"))


class SpectrumBatcher(TaskExecutionSequence):
    '''Break a big list of scans into precursor mass batched blocks of
    spectrum groups with an approximate maximum size. Feeds raw workloads
    into the pipeline.
    '''
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
    '''Wrap scan groups from :class:`SpectrumBatcher` in :class:`StructureMapper` tasks
    and ships them to an appropriate work queue (usually another process).

    .. note::
        The StructureMapper could be applied with or without a database bound to it,
        and for an IPC consumer the database should not be bound, only labeled.
    '''
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

    def execute_task(self, task):
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
class StructureMapper(TaskExecutionSequence):
    """Map spectra against the database using a precursor filtering search strategy,
    generating a task graph of spectrum-structure-mass_shift relationships, a
    :class:`~.WorkloadManager` instance.
    """
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

    def bind_scans(self, source):
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
        predictive_search = self.predictive_search
        hits = predictive_search.peptide_glycosylator._cache_hit
        misses = predictive_search.peptide_glycosylator._cache_miss
        total = hits + misses
        if total > 0:
            self.log("... Cache Performance: %d / %d (%0.2f%%)" % (hits, total, hits / float(total) * 100.0))

    def _prepare_scan(self, scan):
        try:
            return scan.convert()
        except AttributeError:
            if isinstance(scan, ProcessedScan):
                return scan
            else:
                raise

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
            if i % 25 == 0 and i != 0:
                self.log('... Mapped Group %d (%0.2f%%) %0.3f-%0.3f with %d Items (%d Total)' % (
                    i + self.group_i, i * 100.0 / len(self.chunk), lo, hi,
                    solutions.total_work_required(), total_work))
            workload.update(solutions)
        end = time.time()
        self.log("... Mapping Completed (%0.2f sec.)" % (end - start))
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
        workload.pack()
        self.add_decoy_glycans(workload)
        self.predictive_search.construct_peptide_groups(workload)
        return workload


class MapperExecutor(TaskExecutionSequence):
    """This task executor consumes batches of precursor mass-grouped spectra,
    and produces batches of glycopeptides matched to spectra.

    Its task type is :class:`StructureMapper`

    """

    def __init__(self, predictive_searchers, scan_loader, in_queue, out_queue, in_done_event):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()
        self.scan_loader = scan_loader
        self.predictive_searchers = predictive_searchers

    def execute_task(self, mapper_task):
        self.scan_loader.reset()
        # In case this came from a labeled batch mapper and not
        # attached to an actual dynamic glycopeptide generator
        label = mapper_task.predictive_search
        if isinstance(label, str):
            mapper_task.predictive_search = self.predictive_searchers[label]
        mapper_task.bind_scans(self.scan_loader)
        workload = mapper_task()
        self.scan_loader.reset()
        matcher_task = SpectrumMatcher(
            workload, mapper_task.group_i, mapper_task.group_n)
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


class SerializingMapperExecutor(MapperExecutor):
    """This task extends :class:`MapperExecutor` to also serialize its mapping to gzipped
    XML.

    """
    process_name = 'glycopeptide-db-map'

    def __init__(self, predictive_searchers, scan_loader, in_queue, out_queue,
                 in_done_event, tracking_directory=None):
        super(SerializingMapperExecutor, self).__init__(
            predictive_searchers, scan_loader,
            in_queue, out_queue, in_done_event)
        self.predictive_searchers = predictive_searchers
        self.tracking_directory = tracking_directory

    def execute_task(self, mapper_task):
        self.scan_loader.reset()
        label = mapper_task.predictive_search
        mapper_task.predictive_search = self.predictive_searchers[label]
        if debug_mode:
            self.log("... Running %s Mapping with Mass Shifts %r" % (label, mapper_task.mass_shifts))
        mapper_task.bind_scans(self.scan_loader)
        workload = mapper_task()
        self.scan_loader.reset()
        workload.pack()
        start_serializing_workload = time.time()
        workload = serialize_workload(workload)
        end_serializing_workload = time.time()
        self.log("... Serializing Workload Took %0.2f sec." % (
            end_serializing_workload - start_serializing_workload))
        matcher_task = SpectrumMatcher(
            workload, mapper_task.group_i, mapper_task.group_n)

        matcher_task.label = label
        matcher_task.group_i = mapper_task.group_i
        matcher_task.group_n = mapper_task.group_n
        if self.tracking_directory is not None:
            if not os.path.exists(self.tracking_directory):
                os.mkdir(self.tracking_directory)
            track_file = os.path.join(
                self.tracking_directory, "%s_%s_%s.xml" % (
                    label, mapper_task.group_i, mapper_task.group_n))
            with open(track_file, 'wb') as fh:
                fh.write(workload)
        return matcher_task

    def run(self):
        self.try_set_process_name()
        return super(SerializingMapperExecutor, self).run()


@IsTask
class SpectrumMatcher(TaskExecutionSequence):
    """Actually execute the spectrum matching specified in a :class:`~.WorkloadManager`.

    .. note::
        This task may spin up additional processes if :attr:`n_processes` is greater than
        1, but it must be ~4 or better usually to have an appreciable speedup compared to
        a executing the matching in serial. IPC communication is expensive, no matter what.
    """
    def __init__(self, workload, group_i, group_n, scorer_type=None,
                 ipc_manager=None, n_processes=6, mass_shifts=None,
                 evaluation_kwargs=None, cache_seeds=None, **kwargs):
        if scorer_type is None:
            scorer_type = LogIntensityScorer
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
        self.workload = workload
        self.group_i = group_i
        self.group_n = group_n

        self.mass_shifts = mass_shifts
        self.scorer_type = scorer_type
        self.evaluation_kwargs = evaluation_kwargs
        self.evaluation_kwargs.update(kwargs)

        self.ipc_manager = ipc_manager
        self.n_processes = n_processes
        self.cache_seeds = cache_seeds

    def score_spectra(self):
        matcher = MultiScoreGlycopeptideMatcher(
            [], self.scorer_type, None, Parser,
            ipc_manager=self.ipc_manager,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            cache_seeds=self.cache_seeds)

        target_solutions = []
        self.log("... %0.2f%%" % (max((self.group_i - 1), 0) * 100.0 / self.group_n), self.workload)
        lo, hi = self.workload.mass_range()
        self.log("... Query Mass Range: %0.2f-%0.2f" % (lo, hi))

        batches = list(self.workload.batches(matcher.batch_size))
        running_total_work = 0
        total_work = self.workload.total_work_required()
        self.workload.clear()
        for i, batch in enumerate(batches):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work + batch.batch_size, total_work,
                ((running_total_work + batch.batch_size) * 100.) / float(total_work + 1)))
            running_total_work += batch.batch_size
            target_scan_solution_map = matcher._evaluate_hit_groups(
                batch, **self.evaluation_kwargs)
            temp = matcher._collect_scan_solutions(
                target_scan_solution_map, batch.scan_map)
            temp = [case for case in temp if len(case) > 0]
            for case in temp:
                case.simplify()
                # Don't run the select top filters for consistency? They seemed to
                # influence reproducibility. Possibly because glycan decoys get dropped?
                # case.select_top()
            target_solutions.extend(temp)
            batch.clear()
        self.log("... Finished %0.2f%% (%0.2f-%0.2f)" % (max((self.group_i - 1), 0)
                                           * 100.0 / self.group_n, lo, hi))
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

    def configure_task(self, matcher_task):
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
                matcher_task = self.in_queue.get(True, 3)
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


class WorkloadUnpackingMatcherExecutor(MatcherExecutor):
    """This task executor extends :class:`MatcherExecutor` by also deserializing the
    mapping from gzipped XML packaging.

    This type complements :class:`SerializingMapperExecutor`

    """
    def __init__(self, scan_loader, in_queue, out_queue, in_done_event, scorer_type=None,
                 ipc_manager=None, n_processes=6, mass_shifts=None, evaluation_kwargs=None,
                 cache_seeds=None, **kwargs):
        super(WorkloadUnpackingMatcherExecutor, self).__init__(
            in_queue, out_queue, in_done_event, scorer_type, ipc_manager,
            n_processes, mass_shifts, evaluation_kwargs, cache_seeds=cache_seeds, **kwargs)
        self.scan_loader = scan_loader

    def execute_task(self, matcher_task):
        workload = matcher_task.workload
        start = time.time()
        matcher_task.workload = deserialize_workload(
            workload,
            self.scan_loader)
        end = time.time()
        self.log("... Deserializing Workload Took %0.2f sec." % (end - start, ))
        return super(WorkloadUnpackingMatcherExecutor, self).execute_task(matcher_task)


class MappingSerializer(TaskExecutionSequence):
    def __init__(self, storage_directory, in_queue, in_done_event):
        self.storage_directory = storage_directory
        self.in_queue = in_queue
        self.in_done_event = in_done_event
        self.done_event = self._make_event()

    def run(self):
        if not os.path.exists(self.storage_directory):
            os.makedirs(self.storage_directory)
        has_work = True
        while has_work and not self.error_occurred():
            try:
                package = self.in_queue.get(True, 5)
                data = package.workload
                label = package.label
                name = '%s_%s.xml.gz' % (label, package.group_i)
                with open(os.path.join(self.storage_directory, name), 'wb') as fh:
                    fh.write(data)
            except Empty:
                if self.in_done_event.is_set():
                    has_work = False
                    break
        self.done_event.set()
