from collections import defaultdict, namedtuple
from threading import Thread
from multiprocessing import Process, Queue, Event, Manager


try:
    from Queue import Empty as QueueEmptyException
except ImportError:
    from queue import Empty as QueueEmptyException

from ms_deisotope import DeconvolutedPeakSet, isotopic_shift

from glycan_profiling.task import TaskBase
from glycan_profiling.structure import (
    ScanWrapperBase)
from glycan_profiling.chromatogram_tree import Unmodified
from .ref import TargetReference, SpectrumReference
from .spectrum_graph import ScanGraph


neutron_offset = isotopic_shift()


def group_by_precursor_mass(scans, window_size=1.5e-5):
    scans = sorted(
        scans, key=lambda x: x.precursor_information.extracted_neutral_mass,
        reverse=True)
    groups = []
    if len(scans) == 0:
        return groups
    current_group = [scans[0]]
    last_scan = scans[0]
    for scan in scans[1:]:
        delta = abs(
            (scan.precursor_information.extracted_neutral_mass -
             last_scan.precursor_information.extracted_neutral_mass
             ) / last_scan.precursor_information.extracted_neutral_mass)
        if delta > window_size:
            groups.append(current_group)
            current_group = [scan]
        else:
            current_group.append(scan)
        last_scan = scan
    groups.append(current_group)
    return groups


class SpectrumMatchBase(ScanWrapperBase):
    __slots__ = ['scan', 'target', "_mass_shift"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.target = target
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def mass_shift(self):
        return self._mass_shift

    @mass_shift.setter
    def mass_shift(self, value):
        self._mass_shift = value

    @staticmethod
    def threshold_peaks(deconvoluted_peak_set, threshold_fn=lambda peak: True):
        deconvoluted_peak_set = DeconvolutedPeakSet([
            p for p in deconvoluted_peak_set
            if threshold_fn(p)
        ])
        deconvoluted_peak_set._reindex()
        return deconvoluted_peak_set

    def precursor_mass_accuracy(self, offset=0):
        observed = self.precursor_ion_mass
        theoretical = self.target.total_composition().mass + (
            offset * neutron_offset) + self.mass_shift.mass
        return (observed - theoretical) / theoretical

    def determine_precursor_offset(self, probing_range=3):
        best_offset = 0
        best_error = float('inf')
        for i in range(probing_range + 1):
            error = abs(self.precursor_mass_accuracy(i))
            if error < best_error:
                best_error = error
                best_offset = i
        return best_offset

    def __reduce__(self):
        return self.__class__, (self.scan, self.target)

    def get_top_solutions(self):
        return [self]

    def __eq__(self, other):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        try:
            other_target_id = self.target.id
        except AttributeError:
            other_target_id = None
        return (self.scan == other.scan) and (self.target == other.target) and (
            target_id == other_target_id)

    def __hash__(self):
        try:
            target_id = self.target.id
        except AttributeError:
            target_id = None
        return hash((self.scan.id, self.target, target_id))


class SpectrumMatcherBase(SpectrumMatchBase):
    __slots__ = ["spectrum", "_score"]

    def __init__(self, scan, target, mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        self.scan = scan
        self.spectrum = scan.deconvoluted_peak_set
        self.target = target
        self._score = 0
        self._mass_shift = None
        self.mass_shift = mass_shift

    @property
    def score(self):
        return self._score

    def match(self, *args, **kwargs):
        raise NotImplementedError()

    def calculate_score(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        inst = cls(scan, target, mass_shift=mass_shift)
        inst.match(*args, **kwargs)
        inst.calculate_score(*args, **kwargs)
        return inst

    def __getstate__(self):
        return (self.score,)

    def __setstate__(self, state):
        self.score = state[0]

    def __reduce__(self):
        return self.__class__, (self.scan, self.target,)

    @staticmethod
    def load_peaks(scan):
        try:
            return scan.convert(fitted=False, deconvoluted=True)
        except AttributeError:
            return scan

    def __repr__(self):
        return "{self.__class__.__name__}({self.spectrum}, {self.target}, {self.score})".format(
            self=self)

    def plot(self, ax=None, **kwargs):
        from glycan_profiling.plotting import spectral_annotation
        art = spectral_annotation.SpectrumMatchAnnotator(self, ax=ax)
        art.draw(**kwargs)
        return art


class DeconvolutingSpectrumMatcherBase(SpectrumMatcherBase):

    @staticmethod
    def load_peaks(scan):
        try:
            return scan.convert(fitted=True, deconvoluted=False)
        except AttributeError:
            return scan

    def __init__(self, scan, target):
        super(DeconvolutingSpectrumMatcherBase, self).__init__(scan, target)
        self.spectrum = scan.peak_set


class SpectrumMatch(SpectrumMatchBase):

    __slots__ = ['score', 'best_match', 'data_bundle', "q_value", 'id']

    def __init__(self, scan, target, score, best_match=False, data_bundle=None,
                 q_value=None, id=None, mass_shift=None):
        if data_bundle is None:
            data_bundle = dict()

        super(SpectrumMatch, self).__init__(scan, target, mass_shift)

        self.score = score
        self.best_match = best_match
        self.data_bundle = data_bundle
        self.q_value = q_value
        self.id = id

    def clear_caches(self):
        try:
            self.target.clear_caches()
        except AttributeError:
            pass

    def __reduce__(self):
        return self.__class__, (self.scan, self.target, self.score, self.best_match,
                                self.data_bundle, self.q_value, self.id, self.mass_shift)

    def evaluate(self, scorer_type, *args, **kwargs):
        if isinstance(self.scan, SpectrumReference):
            raise TypeError("Cannot evaluate a spectrum reference")
        elif isinstance(self.target, TargetReference):
            raise TypeError("Cannot evaluate a target reference")
        return scorer_type.evaluate(self.scan, self.target, *args, **kwargs)

    def __repr__(self):
        return "SpectrumMatch(%s, %s, %0.4f)" % (self.scan, self.target, self.score)

    @classmethod
    def from_match_solution(cls, match):
        return cls(match.scan, match.target, match.score)


class SpectrumSolutionSet(ScanWrapperBase):

    def __init__(self, scan, solutions):
        self.scan = scan
        self.solutions = solutions
        self.mean = self._score_mean()
        self.variance = self._score_variance()
        self._is_simplified = False
        self._is_top_only = False
        self._target_map = None

    def _invalidate(self):
        self._target_map = None

    @property
    def score(self):
        return self.best_solution().score

    def _make_target_map(self):
        self._target_map = {
            sol.target: sol for sol in self
        }

    def solution_for(self, target):
        if self._target_map is None:
            self._make_target_map()
        return self._target_map[target]

    def precursor_mass_accuracy(self):
        return self.best_solution().precursor_mass_accuracy()

    def best_solution(self):
        return self.solutions[0]

    def _score_mean(self):
        i = 0
        total = 0
        for match in self:
            total += match.score
            i += 1.
        if i > 0:
            return total / i
        else:
            return 0

    def _score_variance(self):
        total = 0.
        i = 0.
        mean = self.mean
        for match in self:
            total += (match.score - mean) ** 2
            i += 1.
        if i < 3:
            return 0
        return total / (i - 2.)

    def __repr__(self):
        if len(self) == 0:
            return "SpectrumSolutionSet(%s, [])" % (self.scan,)
        return "SpectrumSolutionSet(%s, %s, %f)" % (
            self.scan, self.best_solution().target, self.best_solution().score)

    def __getitem__(self, i):
        return self.solutions[i]

    def __iter__(self):
        return iter(self.solutions)

    def __len__(self):
        return len(self.solutions)

    def threshold(self):
        if len(self) == 0:
            return self
        thresh = min(self.mean / 2., self.score / 2.)
        self.solutions = [
            x for x in self if x.score >= thresh
        ]
        self._invalidate()
        return self

    def simplify(self):
        if self._is_simplified:
            return
        self.scan = SpectrumReference(
            self.scan.id, self.scan.precursor_information)
        solutions = []
        best_score = self.best_solution().score
        for sol in self.solutions:
            sm = SpectrumMatch.from_match_solution(sol)
            if abs(sm.score - best_score) < 1e-6:
                sm.best_match = True
            sm.scan = self.scan
            solutions.append(sm)
        self.solutions = solutions
        self._is_simplified = True
        self._invalidate()

    def get_top_solutions(self):
        score = self.best_solution().score
        return [x for x in self.solutions if abs(x.score - score) < 1e-6]

    def select_top(self):
        if self._is_top_only:
            return
        self.solutions = self.get_top_solutions()
        self._is_top_only = True
        self._invalidate()


class DatabaseSearchTrackingMetadata(object):
    def __init__(self, storage=None):
        if storage is None:
            storage = defaultdict(dict)
        self.storage = storage

    def __getitem__(self, key):
        return self.storage[key]

    def __setitem__(self, key, value):
        self.storage[key] = value

    def no_precursor_matches(self, neutral_mass):
        record = self['no_precursor_matches']
        record[neutral_mass] = 1 + record.get(neutral_mass, 0)


class WorkloadManager(object):
    def __init__(self):
        self.scan_map = dict()
        self.hit_map = dict()
        self.hit_to_scan_map = defaultdict(list)
        self.scan_to_hit_map = defaultdict(list)
        self.scan_hit_type_map = defaultdict(lambda: Unmodified.name)

    def add_scan(self, scan):
        self.scan_map[scan.id] = scan

    def add_scan_hit(self, scan, hit, hit_type=Unmodified.name):
        self.hit_map[hit.id] = hit
        self.hit_to_scan_map[hit.id].append(scan)
        self.scan_to_hit_map[scan.id].append(hit.id)
        self.scan_hit_type_map[scan.id, hit.id] = hit_type

    def build_scan_graph(self):
        graph = ScanGraph(self.scan_map.values())
        for key, values in self.hit_to_scan_map.items():
            values = sorted(values, key=lambda x: x.index)
            for i, scan in enumerate(values):
                for other in values[i + 1:]:
                    graph.add_edge_between(scan.id, other.id, key)
        return graph

    def compute_workloads(self):
        graph = self.build_scan_graph()
        components = graph.connected_components()
        workloads = []
        for m in components:
            workloads.append((m, sum([len(c.edges) for c in m])))
        return workloads

    def log_workloads(self, handle, workloads=None):
        if workloads is None:
            workloads = self.compute_workloads()
        workloads = sorted(workloads, key=lambda x: x[0][0].precursor_information.neutral_mass,
                           reverse=True)
        for scans, load in workloads:
            handle.write("Work: %d Comparisons\n" % (load,))
            for scan_node in scans:
                handle.write("%s\t%f\t%d\n" % (
                    scan_node.id,
                    scan_node.precursor_information.neutral_mass,
                    len(scan_node.edges)))

    def __iter__(self):
        yield self.scan_map
        yield self.hit_map
        yield self.hit_to_scan_map

    def __repr__(self):
        template = 'WorkloadManager(scan_map_size={}, hit_map_size={}, total_cross_product={})'
        rendered = template.format(
            len(self.scan_map), len(self.hit_map),
            sum(map(len, self.scan_to_hit_map.values())))
        return rendered

    _workload_batch = staticmethod(
        namedtuple("WorkloadBatch",
                   ['batch_size', 'scan_map',
                    'hit_map', 'hit_to_scan_map',
                    'scan_hit_type_map']))

    def batches(self, max_size=1e5):
        current_batch_size = 0
        current_scan_map = dict()
        current_hit_map = dict()
        current_hit_to_scan_map = defaultdict(list)
        current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)

        source = sorted(
            self.scan_map.items(),
            key=lambda x: x[1].precursor_information.neutral_mass)

        for scan_id, scan in source:
            current_scan_map[scan_id] = scan
            for hit_id in self.scan_to_hit_map[scan_id]:
                current_hit_map[hit_id] = self.hit_map[hit_id]
                current_hit_to_scan_map[hit_id].append(scan_id)
                current_scan_hit_type_map[
                    scan_id, hit_id] = self.scan_hit_type_map[scan_id, hit_id]
                current_batch_size += 1

            if current_batch_size > max_size:
                batch = self._workload_batch(
                    current_batch_size, current_scan_map,
                    current_hit_map, current_hit_to_scan_map,
                    current_scan_hit_type_map)
                current_batch_size = 0
                current_scan_map = dict()
                current_hit_map = dict()
                current_hit_to_scan_map = defaultdict(list)
                current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)
                yield batch

        if current_batch_size > 0:
            batch = self._workload_batch(
                current_batch_size, current_scan_map,
                current_hit_map, current_hit_to_scan_map,
                current_scan_hit_type_map)
            yield batch


class TandemClusterEvaluatorBase(TaskBase):

    neutron_offset = isotopic_shift()

    def __init__(self, tandem_cluster, scorer_type, structure_database, verbose=False,
                 n_processes=1, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.verbose = verbose
        self.n_processes = n_processes
        self.ipc_manager = ipc_manager
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.mass_shifts = mass_shifts

    def search_database_for_precursors(self, mass, precursor_error_tolerance=1e-5):
        return self.structure_database.search_mass_ppm(mass, precursor_error_tolerance)

    def find_precursor_candidates(self, scan, error_tolerance=1e-5, probing_range=0,
                                  mass_shift=None):
        if mass_shift is None:
            mass_shift = Unmodified
        hits = []
        intact_mass = scan.precursor_information.extracted_neutral_mass
        for i in range(probing_range + 1):
            query_mass = intact_mass - (i * self.neutron_offset) - mass_shift.mass
            hits.extend(self.search_database_for_precursors(query_mass, error_tolerance))
        return hits

    def score_one(self, scan, precursor_error_tolerance=1e-5, mass_shifts=None, **kwargs):
        if mass_shifts is None:
            mass_shifts = (Unmodified,)
        solutions = []

        if not scan.precursor_information.defaulted:
            probe = 0
        else:
            probe = self.probing_range_for_missing_precursors
        hits = []
        for mass_shift in mass_shifts:
            hits.extend(self.find_precursor_candidates(
                scan, precursor_error_tolerance, probing_range=probe,
                mass_shift=mass_shift))
            for structure in hits:
                result = self.evaluate(
                    scan, structure, mass_shift=mass_shift, **kwargs)
                solutions.append(result)
        out = SpectrumSolutionSet(
            scan, sorted(
                solutions, key=lambda x: x.score, reverse=True)).threshold()
        return out

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, **kwargs):
        out = []
        for mass_shift in self.mass_shifts:
            # make iterable for score_one API
            mass_shift = (mass_shift,)
            for scan in self.tandem_cluster:
                solutions = self.score_one(
                    scan, precursor_error_tolerance,
                    mass_shifts=mass_shift,
                    **kwargs)
                if len(solutions) > 0:
                    out.append(solutions)
        if simplify:
            for case in out:
                case.simplify()
                case.select_top()
        return out

    def evaluate(self, scan, structure, *args, **kwargs):
        raise NotImplementedError()

    def _map_scans_to_hits(self, scans, precursor_error_tolerance=1e-5):
        groups = group_by_precursor_mass(
            scans, precursor_error_tolerance * 1.5)

        workload = WorkloadManager()

        i = 0
        n = len(scans)
        report_interval = 0.1 * n
        last_report = report_interval
        self.log("... Begin Collecting Hits")
        for mass_shift in self.mass_shifts:
            self.log("... Mapping For %s" % (mass_shift.name,))
            i = 0
            last_report = report_interval
            for group in groups:
                if len(group) == 0:
                    continue
                i += len(group)
                report = False
                if i > last_report:
                    report = True
                    self.log(
                        "... Mapping %0.2f%% of spectra (%d/%d) %0.4f" % (
                            i * 100. / n, i, n,
                            group[0].precursor_information.extracted_neutral_mass))
                    while last_report < i and report_interval != 0:
                        last_report += report_interval
                j = 0
                for scan in group:
                    # scan_map[scan.id] = scan
                    workload.add_scan(scan)

                    # For a sufficiently dense database or large value of probe, this
                    # could easily throw the mass interval cache scheme into hysteresis.
                    # If this does occur, instead of doing this here, we could search all
                    # defaulted precursors afterwards.
                    if not scan.precursor_information.defaulted:
                        probe = 0
                    else:
                        probe = self.probing_range_for_missing_precursors
                    hits = self.find_precursor_candidates(
                        scan, precursor_error_tolerance, probing_range=probe,
                        mass_shift=mass_shift)
                    for hit in hits:
                        j += 1
                        workload.add_scan_hit(scan, hit, mass_shift.name)
                        # hit_to_scan[hit.id].append(scan)
                        # hit_map[hit.id] = hit
                if report:
                    self.log("... Mapping Segment Done. (%d spectrum-pairs)" % (j,))
        # return scan_map, hit_map, hit_to_scan
        self.log("... Computing Workload Graph")
        try:
            with open("worklog.txt", 'a') as f:
                f.write("\n#GENERATION\n%s\n" % (workload,))
                workload.log_workloads(f)
        except IOError as e:
            self.error("An error occurred while trying to write workload log", e)
        self.log("... Workload Graph Traversed")
        return workload

    def _evaluate_hit_groups_single_process(self, workload, *args, **kwargs):
        batch_size, scan_map, hit_map, hit_to_scan, scan_hit_type_map = workload
        scan_solution_map = defaultdict(list)
        self.log("... Searching Hits (%d:%d)" % (
            len(hit_to_scan),
            sum(map(len, hit_to_scan.values())))
        )
        i = 0
        n = len(hit_to_scan)
        for hit_id, scan_list in hit_to_scan.items():
            i += 1
            if i % 1000 == 0:
                self.log("... %0.2f%% of Hits Searched (%d/%d)" %
                         (i * 100. / n, i, n))
            hit = hit_map[hit_id]
            solutions = []
            for scan_id in scan_list:
                scan = scan_map[scan_id]
                match = SpectrumMatch.from_match_solution(
                    self.evaluate(scan, hit, *args, **kwargs))
                scan_solution_map[scan.id].append(match)
                solutions.append(match)
            # Assumes all matches to the same target structure share a cache
            match.clear_caches()
            self.reset_parser()

        return scan_solution_map

    def _collect_scan_solutions(self, scan_solution_map, scan_map):
        result_set = []
        for scan_id, solutions in scan_solution_map.items():
            if len(solutions) == 0:
                continue
            scan = scan_map[scan_id]
            out = SpectrumSolutionSet(scan, sorted(
                solutions, key=lambda x: x.score, reverse=True)).threshold()
            if len(out) == 0:
                continue
            result_set.append(out)
        return result_set

    @property
    def _worker_specification(self):
        raise NotImplementedError()

    def _evaluate_hit_groups_multiple_processes(self, workload, **kwargs):
        batch_size, scan_map, hit_map, hit_to_scan, scan_hit_type_map = workload
        worker_type, init_args = self._worker_specification
        dispatcher = IdentificationProcessDispatcher(
            worker_type, self.scorer_type, evaluation_args=kwargs, init_args=init_args,
            n_processes=self.n_processes, ipc_manager=self.ipc_manager, mass_shift_map={
                m.name: m for m in self.mass_shifts
            })
        return dispatcher.process(scan_map, hit_map, hit_to_scan, scan_hit_type_map)

    def _evaluate_hit_groups(self, batch, **kwargs):
        if self.n_processes == 1 or len(batch.hit_map) < 500:
            return self._evaluate_hit_groups_single_process(
                batch, **kwargs)
        else:
            return self._evaluate_hit_groups_multiple_processes(
                batch, **kwargs)

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, **kwargs):
        workload = self._map_scans_to_hits(
            scans, precursor_error_tolerance)
        solutions = []
        for batch in workload.batches():
            scan_solution_map = self._evaluate_hit_groups(batch, **kwargs)
            solutions += self._collect_scan_solutions(scan_solution_map, batch.scan_map)
        return solutions


class IdentificationProcessDispatcher(TaskBase):
    """Orchestrates distributing the spectrum match evaluation
    task across several processes.

    The distribution pushes individual structures ("targets") and
    the scan ids they mapped to in the MS1 dimension to each worker.

    All scans in the batch being worked on are made available over
    an IPC synchronized dictionary.

    Attributes
    ----------
    done_event : multiprocessing.Event
        An Event indicating that all work items
        have been placed on `input_queue`
    evaluation_args : dict
        A dictionary containing arguments to be
        passed through to `evaluate` on the worker
        processes.
    init_args : dict
        A dictionary containing extra arguments
        to use when initializing worker process
        instances.
    input_queue : multiprocessing.Queue
        The queue from which worker processes
        will read their targets and scan id mappings
    ipc_manager : multiprocessing.SyncManager
        Provider of IPC dictionary synchronization
    log_controller : MessageSpooler
        Logging facility to funnel messages from workers
        through into the main process's log stream
    n_processes : int
        The number of worker processes to spawn
    output_queue : multiprocessing.Queue
        The queue which worker processes will
        put ther results on, read in the main
        process.
    scan_load_map : multiprocessing.SyncManager.dict
        An inter-process synchronized dictionary which
        maps scan ids to scans. Used by worker processes
        to request individual scans by name when they are
        not found locally.
    scan_solution_map : defaultdict(list)
        A mapping from scan id to all candidate solutions.
    scorer_type : SpectrumMatcherBase
        The type used by workers to evaluate spectrum matches
    worker_type : SpectrumIdentificationWorkerBase
        The type instantiated to construct worker processes
    workers : list
        Container for created workers.
    """
    def __init__(self, worker_type, scorer_type, evaluation_args=None, init_args=None,
                 mass_shift_map=None, n_processes=3, ipc_manager=None):
        if ipc_manager is None:
            self.log("Creating IPC Manager. Prefer to pass a reusable IPC Manager instead.")
            ipc_manager = Manager()
        if evaluation_args is None:
            evaluation_args = dict()
        if mass_shift_map is None:
            mass_shift_map = {
                Unmodified.name: Unmodified
            }

        self.worker_type = worker_type
        self.scorer_type = scorer_type
        self.n_processes = n_processes
        self.done_event = Event()
        self.input_queue = Queue(1000)
        self.output_queue = Queue(1000)
        self.scan_solution_map = defaultdict(list)
        self.evaluation_args = evaluation_args
        self.init_args = init_args
        self.workers = []
        self.log_controller = self.ipc_logger()
        self.ipc_manager = ipc_manager
        self.local_scan_map = dict()
        self.scan_load_map = self.ipc_manager.dict()
        self.local_mass_shift_map = mass_shift_map
        self.mass_shift_load_map = self.ipc_manager.dict(mass_shift_map)

    def clear_pool(self):
        # self.log("... Clearing Worker Pool")
        self.scan_load_map.clear()
        self.local_scan_map.clear()
        self.ipc_manager = None
        for worker in self.workers:
            try:
                worker.join()
            except AttributeError:
                pass

    def create_pool(self, scan_map):
        self.scan_load_map.clear()
        self.scan_load_map.update(scan_map)
        self.local_scan_map.clear()
        self.local_scan_map.update(scan_map)

        for i in range(self.n_processes):
            worker = self.worker_type(
                self.input_queue, self.output_queue, self.done_event,
                self.scorer_type, self.evaluation_args,
                self.scan_load_map, self.mass_shift_load_map,
                log_handler=self.log_controller.sender(), **self.init_args)
            worker.start()
            self.workers.append(worker)

    def all_workers_finished(self):
        return all([worker.all_work_done() for worker in self.workers])

    def feeder(self, hit_map, hit_to_scan, scan_hit_type_map):
        i = 0
        n = len(hit_to_scan)
        seen = dict()
        for hit_id, scan_ids in hit_to_scan.items():
            i += 1
            hit = hit_map[hit_id]
            # This sanity checking is likely unnecessary, and is a hold-over from
            # debugging redundancy in the result queue. For the moment, it is retained
            # to catch "new" bugs.
            # If a hit structure's id doesn't match the id it was looked up with, something
            # may be wrong with the upstream process. Log this event.
            if hit.id != hit_id:
                self.log("Hit %r doesn't match its id %r" % (hit, hit_id))
                if hit_to_scan[hit.id] != scan_ids:
                    self.log("Mismatch leads to different scans! (%d, %d)" % (
                        len(scan_ids), len(hit_to_scan[hit.id])))
            # If a hit structure has been seen multiple times independent of whether or
            # not the expected hit id matches, something may be wrong in the upstream process.
            # Log this event.
            if hit.id in seen:
                self.log("Hit %r already dealt under hit_id %r, now again at %r" % (
                    hit, seen[hit.id], hit_id))
            seen[hit.id] = hit_id

            try:
                work_order = self.build_work_order(
                    hit_id, hit_map, scan_hit_type_map, hit_to_scan)
                self.input_queue.put(work_order)
                # Set a long progress update interval because the feeding step is less
                # important than the processing step. Additionally, as the two threads
                # run concurrently, the feeding thread can log a short interval before
                # the entire process has formally logged that it has started.
                if i % 10000 == 0:
                    self.log("...... Dealt %d work items (%0.2f%% Complete)" % (i,
                             i * 100.0 / n))
            except Exception as e:
                self.log("An exception occurred while feeding %r and %d scan ids: %r" % (hit_id, len(scan_ids), e))
        # self.log("...... Finished dealing %d work items" % (i,))
        self.done_event.set()
        return

    def build_work_order(self, hit_id, hit_map, scan_hit_type_map, hit_to_scan):
        return (hit_map[hit_id],
                [(s, scan_hit_type_map[s, hit_id])
                 for s in hit_to_scan[hit_id]])

    def spawn_queue_feeder(self, hit_map, hit_to_scan, scan_hit_type_map):
        feeder_thread = Thread(target=self.feeder, args=(hit_map, hit_to_scan,
                                                         scan_hit_type_map))
        feeder_thread.daemon = True
        feeder_thread.start()
        return feeder_thread

    def _reconstruct_missing_work_items(self, seen, hit_map, hit_to_scan, scan_hit_type_map):
        missing = set(hit_to_scan) - set(seen)
        for missing_id in missing:
            target, scan_spec = self.build_work_order(
                missing_id, hit_map, scan_hit_type_map, hit_to_scan)
            target, score_map = self.evalute_work_order_local(target, scan_spec)
            seen[target.id] = -1
            self.store_result(target, score_map, self.scan_map)

    def store_result(self, target, score_map, scan_map):
        try:
            target.clear_caches()
        except AttributeError:
            pass

        j = 0
        for hit_spec, score in score_map.items():
            scan_id, shift_type = hit_spec
            j += 1
            if j % 1000 == 0:
                self.log("...... Mapping match %d for %s on %s with score %r" % (j, target, scan_id, score))
            self.scan_solution_map[scan_id].append(
                SpectrumMatch(scan_map[scan_id], target, score,
                              mass_shift=self.local_mass_shift_map[shift_type]))

    def fetch_scan(self, key):
        return self.local_scan_map[key]

    def fetch_mass_shift(self, key):
        return self.local_mass_shift_map[key]

    def evalute_work_order_local(self, structure, scan_specification):
        scan_specification = [(self.fetch_scan(i), self.fetch_mass_shift(j)) for i, j in scan_specification]
        solution_target = None
        solution = None
        solution_map = dict()
        for scan, mass_shift in scan_specification:
            solution = self.evaluate(scan, structure, mass_shift=mass_shift,
                                     **self.evaluation_args)
            solution_map[scan.id, mass_shift.name] = solution.score
            solution_target = solution.target
        if solution is not None:
            try:
                solution.target.clear_caches()
            except AttributeError:
                pass
        return solution_target, solution_map

    def evaluate(self, scan, structure, *args, **kwargs):
        if not self.workers:
            raise ValueError(
                "Cannot evaluate a spectrum match without an instantiated worker pool.")
        worker = self.workers[0]
        return worker.evaluate(scan, structure, *args, **kwargs)

    def process(self, scan_map, hit_map, hit_to_scan, scan_hit_type_map):
        feeder_thread = self.spawn_queue_feeder(
            hit_map, hit_to_scan, scan_hit_type_map)
        self.create_pool(scan_map)
        has_work = True
        i = 0
        n = len(hit_to_scan)
        if n != len(hit_map):
            self.log("There is a mismatch between hit_map (%d) and hit_to_scan (%d)" % (
                len(hit_map), n))
        # Track the iteration number a particular structure (id) has been received
        # on. This may be used to detect if a structure has been received multiple
        # times, and to determine when all expected structures have been received.
        seen = dict()
        # Keep a running tally of the number of iterations when there are pending
        # structure matches to process, but all workers claim to be done.
        strikes = 0
        self.log("... Searching Matches (%d)" % (n,))
        while has_work:
            try:
                target, score_map = self.output_queue.get(True, 1)
                if target.id in seen:
                    self.log(
                        "Duplicate Results For %s. First seen at %d, now again at %d" % (
                            target.id, seen[target.id], i))
                else:
                    seen[target.id] = i
                if i > n:
                    self.log(
                        "Warning: %d additional output received. %s and %d matches." % (
                            i - n, target, len(score_map)))

                i += 1
                strikes = 0
                if i % 1000 == 0:
                    self.log(
                        "...... Processed %d matches (%0.2f%%)" % (i, i * 100. / n))
            except QueueEmptyException:
                if self.all_workers_finished():
                    if len(seen) == n:
                        has_work = False
                    else:
                        strikes += 1
                        if strikes % 50 == 0:
                            self.log(
                                "...... %d cycles without output (%d/%d, %0.2f%% Done)" % (
                                    strikes, len(seen), n, len(seen) * 100. / n))
                        if strikes > 1e2:
                            self.log(
                                "...... Too much time has elapsed with"
                                " missing items. Evaluating serially.")
                            self._reconstruct_missing_work_items(
                                seen, hit_map, hit_to_scan, scan_hit_type_map)
                            has_work = False
                else:
                    strikes += 1
                    if strikes % 50 == 0:
                        self.log(
                            "...... %d cycles without output (%d/%d, %0.2f%% Done)" % (
                                strikes, len(seen), n, len(seen) * 100. / n))
                continue
            self.store_result(target, score_map, scan_map)
        self.log("... Finished Processing Matches (%d)" % (i,))
        self.clear_pool()
        self.log_controller.stop()
        feeder_thread.join()
        return self.scan_solution_map


class SpectrumIdentificationWorkerBase(Process):
    def __init__(self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
                 spectrum_map, mass_shift_map, log_handler):
        Process.__init__(self)
        self.daemon = True
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.done_event = done_event
        self.scorer_type = scorer_type
        self.evaluation_args = evaluation_args

        self.spectrum_map = spectrum_map
        self.mass_shift_map = mass_shift_map

        self.local_scan_map = dict()
        self.local_mass_shift_map = dict({
            Unmodified.name: Unmodified
        })
        self.solution_map = dict()
        self._work_complete = Event()
        self.log_handler = log_handler

    def fetch_scan(self, key):
        try:
            return self.local_scan_map[key]
        except KeyError:
            scan = self.spectrum_map[key]
            self.local_scan_map[key] = scan
            return scan

    def fetch_mass_shift(self, key):
        try:
            return self.local_mass_shift_map[key]
        except KeyError:
            mass_shift = self.mass_shift_map[key]
            self.local_mass_shift_map[key] = mass_shift
            return mass_shift

    def all_work_done(self):
        return self._work_complete.is_set()

    def pack_output(self, target):
        if self.solution_map:
            self.output_queue.put((target, self.solution_map))
        self.solution_map = dict()

    def evaluate(self, scan, structure, *args, **kwargs):
        raise NotImplementedError()

    def handle_item(self, structure, scan_specification):
        scan_specification = [(self.fetch_scan(i), self.fetch_mass_shift(j)) for i, j in scan_specification]
        solution_target = None
        solution = None
        for scan, mass_shift in scan_specification:
            solution = self.evaluate(scan, structure, mass_shift=mass_shift,
                                     **self.evaluation_args)
            self.solution_map[scan.id, mass_shift.name] = solution.score
            solution_target = solution.target
        if solution is not None:
            try:
                solution.target.clear_caches()
            except AttributeError:
                pass
        self.pack_output(solution_target)

    def task(self):
        has_work = True
        items_handled = 0
        while has_work:
            try:
                structure, scan_ids = self.input_queue.get(True, 5)
            except QueueEmptyException:
                if self.done_event.is_set():
                    has_work = False
                    break
            items_handled += 1
            self.handle_item(structure, scan_ids)

        self._work_complete.set()

    def run(self):
        try:
            self.task()
        except Exception:
            import traceback
            self.log_handler("An exception occurred while executing %r.\n%s" % (
                self, traceback.format_exc()))
            raise
