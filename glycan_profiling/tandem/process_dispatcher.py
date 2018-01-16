from collections import defaultdict
from threading import Thread
from multiprocessing import Process, Queue, Event, Manager

try:
    from Queue import Empty as QueueEmptyException
except ImportError:
    from queue import Empty as QueueEmptyException

from glypy.utils import uid

from glycan_profiling.task import TaskBase
from glycan_profiling.chromatogram_tree import Unmodified

from .spectrum_match import SpectrumMatch


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
        """Tear down spawned worker processes and clear
        the shared memory server
        """
        self.scan_load_map.clear()
        self.local_scan_map.clear()
        self.ipc_manager = None
        for worker in self.workers:
            try:
                worker.join()
            except AttributeError:
                pass

    def create_pool(self, scan_map):
        """Spawn a pool of workers and a supporting process
        for sharing scans from ``scan_map`` by id with the workers
        so they can load scans on demand.

        Parameters
        ----------
        scan_map : dict
            Map scan id to :class:`.ProcessedScan` object
        """
        self.scan_load_map.clear()
        self.scan_load_map.update(scan_map)
        self.local_scan_map.clear()
        self.local_scan_map.update(scan_map)

        for i in range(self.n_processes):
            worker = self.worker_type(
                input_queue=self.input_queue, output_queue=self.output_queue,
                done_event=self.done_event,
                scorer_type=self.scorer_type,
                evaluation_args=self.evaluation_args,
                spectrum_map=self.scan_load_map,
                mass_shift_map=self.mass_shift_load_map,
                log_handler=self.log_controller.sender(), **self.init_args)
            worker.start()
            self.workers.append(worker)

    def all_workers_finished(self):
        return all([worker.all_work_done() for worker in self.workers])

    def feeder(self, hit_map, hit_to_scan, scan_hit_type_map):
        """Push tasks onto the input queue feeding the worker
        processes.

        Parameters
        ----------
        hit_map : dict
            Maps hit id to structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match
        """
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
        """Packs several task-defining data structures into a simple to unpack payload for
        sending over IPC to worker processes.

        Parameters
        ----------
        hit_id : int
            The id number of a hit structure
        hit_map : dict
            Maps hit_id to hit structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match

        Returns
        -------
        tuple
            Packaged message payload
        """
        return (hit_map[hit_id],
                [(s, scan_hit_type_map[s, hit_id])
                 for s in hit_to_scan[hit_id]])

    def spawn_queue_feeder(self, hit_map, hit_to_scan, scan_hit_type_map):
        """Create a thread to run :meth:`feeder` with the provided arguments
        so that work can be sent in tandem with waiting for results

        Parameters
        ----------
        hit_map : dict
            Maps hit id to structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match

        Returns
        -------
        Thread
        """
        feeder_thread = Thread(target=self.feeder, args=(hit_map, hit_to_scan,
                                                         scan_hit_type_map))
        feeder_thread.daemon = True
        feeder_thread.start()
        return feeder_thread

    def _reconstruct_missing_work_items(self, seen, hit_map, hit_to_scan, scan_hit_type_map):
        """Handle task items that are pending after it is believed that the workers
        have crashed or the network communication has failed.

        Parameters
        ----------
        seen : dict
            Map of hit ids that have already been handled
        hit_map : dict
            Maps hit_id to hit structure
        hit_to_scan : dict
            Maps hit id to list of scan ids
        scan_hit_type_map : dict
            Maps (hit id, scan id) to the type of mass shift
            applied for this match
        """
        missing = set(hit_to_scan) - set(seen)
        for missing_id in missing:
            target, scan_spec = self.build_work_order(
                missing_id, hit_map, scan_hit_type_map, hit_to_scan)
            target, score_map = self.evalute_work_order_local(target, scan_spec)
            seen[target.id] = (-1, 0)
            self.store_result(target, score_map, self.local_scan_map)

    def store_result(self, target, score_map, scan_map):
        """Save the spectrum match scores for ``target`` against the
        set of matched scans

        Parameters
        ----------
        target : object
            Structure that was matched
        score_map : dict
            Maps (scan id, mass shift name) to score
        scan_map : dict
            Maps scan id to :class:`.ProcessedScan`
        """
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
        """Mimic the main loop of :class:`SpectrumIdentificationWorkerBase`
        to evaluate task items locally.

        Parameters
        ----------
        structure : object
            object to match against MSn scans
        scan_specification : list
            List of tuples of (scan id, mass shift name) to match against ``structure``

        Returns
        -------
        object
            Fully processed version of ``structure``
        dict
            Mapping of (scan id, mass shift name) to match score
        """
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
        """Evaluate the match between ``structure`` and ``scan``
        using the scoring method provided by a worker process
        locally.

        Parameters
        ----------
        scan : ms_deisotope.ProcessedScan
            MSn scan to match against
        structure : object
            Structure to match against ``scan``
        *args
            Propagated to scoring function
        **kwargs
            Propagated to scoring function

        Returns
        -------
        SpectrumMatcherBase

        Raises
        ------
        ValueError
            If no workers have been created, then we cannot use their
            evaluate method.
        """
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
                target, score_map, token = self.output_queue.get(True, 1)
                if target.id in seen:
                    self.log(
                        "...... Duplicate Results For %s. First seen at %r, now again at %r" % (
                            target.id, seen[target.id], (i, token)))
                else:
                    seen[target.id] = (i, token)
                if (i > n) and ((i - n) % 10 == 0):
                    self.log(
                        "...... Warning: %d additional output received. %s and %d matches." % (
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
        self.token = uid()

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
            self.output_queue.put((target, self.solution_map, self.token))
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
                else:
                    continue
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
