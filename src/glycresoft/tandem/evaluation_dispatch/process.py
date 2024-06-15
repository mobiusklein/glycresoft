import os
import time
import traceback
from typing import Any, List, Mapping, Tuple, Type


try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import deque
from threading import Thread

import multiprocessing
from multiprocessing import Process, Event, Manager, JoinableQueue
from multiprocessing.managers import RemoteError

try:
    from Queue import Empty as QueueEmptyException
except ImportError:
    from queue import Empty as QueueEmptyException

from glypy.utils import uid

from ms_deisotope.data_source import ProcessedScan

from glycresoft.task import TaskBase
from glycresoft.chromatogram_tree import Unmodified, MassShift
from glycresoft.structure import LRUMapping

from .evaluation import SolutionHandler, LocalSpectrumEvaluator, SpectrumEvaluatorBase
from .task import TaskQueueFeeder, DBHit, HitID, GroupID
from .utils import SentinelToken, ProcessDispatcherState


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))


class IdentificationProcessDispatcher(TaskBase):
    """Orchestrates distributing the spectrum match evaluation
    task across several processes.

    The distribution pushes individual structures ("targets") and
    the scan ids they mapped to in the MSn dimension to each worker.

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

    solution_handler: SolutionHandler
    state: ProcessDispatcherState
    worker_type: Type['SpectrumIdentificationWorkerBase']

    input_queue: JoinableQueue
    output_queue: JoinableQueue

    producer_thread_done_event: Event


    post_search_trailing_timeout = 1.5e2
    child_failure_timeout = 2.5e2

    def __init__(self, worker_type, scorer_type, evaluation_args=None, init_args=None,
                 mass_shift_map=None, n_processes=3, ipc_manager=None, solution_handler_type=None):
        if solution_handler_type is None:
            solution_handler_type = SolutionHandler
        if ipc_manager is None:
            self.log("Creating IPC Manager. Prefer to pass a reusable IPC Manager instead.")
            ipc_manager = Manager()
        if evaluation_args is None:
            evaluation_args = dict()
        if mass_shift_map is None:
            mass_shift_map = {
                Unmodified.name: Unmodified
            }

        self.state = ProcessDispatcherState.start
        self.ipc_manager = ipc_manager
        self.worker_type = worker_type
        self.scorer_type = scorer_type
        self.n_processes = n_processes

        self.producer_thread_done_event = self.ipc_manager.Event()
        self.consumer_done_event = self.ipc_manager.Event()

        self.input_queue = self._make_input_queue()
        self.output_queue = self._make_output_queue()

        self.feeder = TaskQueueFeeder(self.input_queue, self.producer_thread_done_event)

        self.solution_handler = solution_handler_type({}, {}, mass_shift_map)

        self.evaluation_args = evaluation_args
        self.init_args = init_args
        self.workers = []
        self.log_controller = self.ipc_logger()
        self.local_scan_map = dict()
        self.scan_load_map = self.ipc_manager.dict()
        self.local_mass_shift_map = mass_shift_map
        self.mass_shift_load_map = self.ipc_manager.dict(mass_shift_map)
        self.structure_map = dict()
        self._token_to_worker = {}
        self._has_received_token = set()
        self._has_remote_error = False
        self._result_buffer = deque()

    @property
    def scan_solution_map(self):
        return self.solution_handler.scan_solution_map

    def _make_input_queue(self):
        try:
            return JoinableQueue(int(1e5))
        except (OSError, ValueError):
            return JoinableQueue()

    def _make_output_queue(self):
        try:
            return JoinableQueue(int(1e7))
        except (OSError, ValueError):
            return JoinableQueue()

    def clear_pool(self):
        """Tear down spawned worker processes and clear
        the shared memory server
        """
        self.scan_load_map.clear()
        self.local_scan_map.clear()
        if self.state in (ProcessDispatcherState.running, ProcessDispatcherState.running_local_workers_dead):
            self.state = ProcessDispatcherState.terminating
        elif self.state == ProcessDispatcherState.running_local_workers_live:
            self.state = ProcessDispatcherState.terminating_workers_live
        else:
            self.state = ProcessDispatcherState.terminating
        for _i, worker in enumerate(self.workers):
            exitcode = worker.exitcode
            if exitcode != 0 and exitcode is not None:
                self.log("... Worker Process %r had exitcode %r" % (worker, exitcode))
            try:
                worker.join(1)
            except AttributeError:
                pass
            if worker.is_alive() and worker.token not in self._has_received_token:
                self.debug("... Worker Process %r is still alive and incomplete" % (worker, ))
                worker.terminate()

    def create_pool(self, scan_map):
        """Spawn a pool of workers and a supporting process
        for sharing scans from ``scan_map`` by id with the workers
        so they can load scans on demand.

        Parameters
        ----------
        scan_map : dict
            Map scan id to :class:`.ProcessedScan` object
        """
        self.state = ProcessDispatcherState.spawning
        self.scan_load_map.clear()
        # Do not copy the complete scan object into the IPC manager dict, as this will
        # pickle the scan source, and then unpickle it several times on the recieving end.
        self.scan_load_map.update({k: pickle.dumps(v.copy(deep=False).unbind(), -1) for k, v in scan_map.items()})
        self.local_scan_map.clear()
        self.local_scan_map.update(scan_map)

        self.input_queue = self._make_input_queue()
        self.output_queue = self._make_output_queue()
        self.feeder = TaskQueueFeeder(self.input_queue, self.producer_thread_done_event)

        for _i in range(self.n_processes):
            worker = self.worker_type(
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                producer_done_event=self.producer_thread_done_event,
                consumer_done_event=self.consumer_done_event,
                scorer_type=self.scorer_type,
                evaluation_args=self.evaluation_args,
                spectrum_map=self.scan_load_map,
                mass_shift_map=self.mass_shift_load_map,
                log_handler=self.log_controller.sender(),
                solution_packer=self.solution_handler.packer,
                **self.init_args)
            worker._work_complete = self.ipc_manager.Event()
            worker.start()
            self._token_to_worker[worker.token] = worker
            self.workers.append(worker)

    def all_workers_finished(self):
        """Check if all worker processes have finished.
        """
        worker_still_busy = False
        i = 0
        j = 0
        for worker in self.workers:
            i += 1
            try:
                is_done = worker.all_work_done()
                j += is_done
                if not is_done:
                    worker_still_busy = True
                    return worker_still_busy
            except (RemoteError, KeyError):
                worker_still_busy = True
                self._has_remote_error = True
        self.log("... All Workers Done: %r (%d/%d), Error? %r" % (not worker_still_busy, j, i, self._has_remote_error))
        return not worker_still_busy

    def build_work_order(self, hit_id: HitID, hit_map: Mapping[HitID, DBHit],
                         scan_hit_type_map: Mapping[Tuple[str, HitID], MassShift],
                         hit_to_scan: Mapping[HitID, List[str]]):
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
        return self.feeder.build_work_order(hit_id, hit_map, scan_hit_type_map, hit_to_scan)

    def spawn_queue_feeder(self, hit_map, hit_to_scan, scan_hit_type_map, hit_to_group=None):
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
        hit_to_group: dict, optional
            Maps group id to the set of hit ids which are
        Returns
        -------
        Thread
        """
        feeder_thread = Thread(
            target=self.feeder, args=(hit_map, hit_to_scan, scan_hit_type_map, hit_to_group))
        feeder_thread.daemon = True
        feeder_thread.start()
        return feeder_thread

    def store_result(self, target, score_map):
        """Save the spectrum match scores for ``target`` against the
        set of matched scans

        Parameters
        ----------
        target : object
            The structure that was matched
        score_map : dict
            Maps (scan id, mass shift name) to score
        scan_map : dict
            Maps scan id to :class:`.ProcessedScan`
        """
        self.solution_handler(target, score_map)

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
        i = 0
        n = len(missing)
        evaluator = LocalSpectrumEvaluator(
            self.workers[0], self.local_scan_map, self.local_mass_shift_map,
            self.solution_handler.packer)
        for missing_id in missing:
            target, scan_spec = self.build_work_order(missing_id, hit_map, scan_hit_type_map, hit_to_scan)
            try:
                # Change this to be the target's ID and look it up in hit_map
                target_id, score_map = evaluator.handle_item(target, scan_spec)
                target = hit_map.get(target_id, target_id)
                seen[target.id] = (-1, 0)
                self.store_result(target, score_map)
                i += 1
                if i % 1000 == 0:
                    self.log("...... Processed %d local matches (%0.2f%%)" % (i, i * 100. / n))
            except Exception as err:
                self.error("...... An error occurred while processing %r: %r" % (target, err), exception=err)
                raise
        return i

    def get_result(self, seen, hit_map):
        if self._result_buffer:
            return self._result_buffer.popleft()

        payload = self.output_queue.get(True, 1)
        self.output_queue.task_done()

            # Change this to be the target's ID and look it up in hit_map
        if isinstance(payload, SentinelToken):
            self.debug("...... Received sentinel from %s" %
                        (self._token_to_worker[payload.token].name))
            self._has_received_token.add(payload.token)
        else:
            self._result_buffer.extend(payload)
            if self._result_buffer:
                return self._result_buffer.popleft()

    def process(self, scan_map: Mapping[str, ProcessedScan],
                hit_map: Mapping[HitID, DBHit],
                hit_to_scan: Mapping[HitID, List[str]],
                scan_hit_type_map: Mapping[Tuple[str, HitID], MassShift],
                hit_group_map: Mapping[GroupID, List[HitID]]=None) -> Mapping[str, List]:
        """Evaluate all spectrum matches, spreading work among the worker
        process pool.

        Parameters
        ----------
        scan_map : :class:`dict`
            Mapping from :attr:`~.ScanBase.id` to :class:`~.ScanBase` instance
        hit_map : :class:`dict`
            Mapping from `structure.id` to the structure type to matche against
        hit_to_scan : :class:`dict`
            Mapping from `structure.id` to a list of :attr:`~.ScanBase.id` which
            it has a precursor mass match.
        scan_hit_type_map : :class:`dict`
            Mapping from (:attr:`~.ScanBase.id`, `structure.id`) to :class:`MassShiftBase`
        hit_group_map : :class:`dict`, optional
            An additional :class:`dict` that maps abstract group ids to sets of hit ids
            in `hit_map` which all share information that can be computed once and shared
            between all members of the group.
        """
        self.structure_map = hit_map
        self.solution_handler.scan_map = scan_map
        self.create_pool(scan_map)

        # Won't feed hit groups until after more work is done here.
        feeder_thread = self.spawn_queue_feeder(
            hit_map, hit_to_scan, scan_hit_type_map, hit_group_map)
        has_work = True
        i = 0
        scan_count = len(scan_map)
        n = len(hit_to_scan)
        if n != len(hit_map):
            self.log("There is a mismatch between hit_map (%d) and hit_to_scan (%d)" % (
                len(hit_map), n))
        n_spectrum_matches = sum(map(len, hit_to_scan.values()))
        # Track the iteration number a particular structure (id) has been received
        # on. This may be used to detect if a structure has been received multiple
        # times, and to determine when all expected structures have been received.
        seen = dict()
        # Keep a running tally of the number of iterations when there are pending
        # structure matches to process, but all workers claim to be done.
        strikes = 0
        self.state = ProcessDispatcherState.running
        self.log("... Searching Matches (%d)" % (n_spectrum_matches,))
        last_log_time = start_time = time.time()
        should_log = False
        log_cycle = 5000
        while has_work:
            try:
                payload = self.get_result(seen, hit_map)
                if payload is None:
                    continue
                else:
                    (target_id, score_map, token) = payload
                target = hit_map.get(target_id, target_id)
                if target.id in seen:
                    self.log(
                        "...... Duplicate Results For %s. First seen at %r, now again at %r" % (
                            target, seen[target.id], (i, token)))
                else:
                    seen[target.id] = (i, token)
                if (i > n) and ((i - n) % 10 == 0):
                    self.log(
                        "...... Warning: %d additional output received. %s and %d matches." % (
                            i - n, target, len(score_map)))

                i += 1
                strikes = 0
                if i % log_cycle == 0 or should_log:
                    last_log_time = time.time()
                    should_log = False
                    self.log(
                        "...... Processed %d structures (%0.2f%%)" % (i, i * 100. / n))
                self.store_result(target, score_map)
            except QueueEmptyException:
                if len(seen) == n:
                    has_work = False
                # do worker life cycle management here
                elif self.all_workers_finished():
                    if len(seen) == n:
                        has_work = False
                    else:
                        strikes += 1
                        if strikes % 50 == 0:
                            self.log(
                                "...... %d cycles without output (%d/%d, %0.2f%% Done)" % (
                                    strikes, len(seen), n, len(seen) * 100. / n))
                        if strikes > self.post_search_trailing_timeout:
                            self.state = ProcessDispatcherState.running_local_workers_dead
                            self.log(
                                "...... Too much time has elapsed with"
                                " missing items. Evaluating serially.")
                            i += self._reconstruct_missing_work_items(
                                seen, hit_map, hit_to_scan, scan_hit_type_map)
                            has_work = False
                            self.debug("...... Processes")
                            for worker in self.workers:
                                self.debug("......... %r" % (worker,))
                            self.debug("...... IPC Manager: %r" % (self.ipc_manager,))
                else:
                    strikes += 1
                    if strikes % 10 == 0:
                        check_time = time.time()
                        if check_time - last_log_time > 10:
                            should_log = True
                    if strikes % 50 == 0:
                        self.log(
                            "...... %d cycles without output (%d/%d, %0.2f%% Done, %d children still alive)" % (
                                strikes, len(seen), n, len(seen) * 100. / n,
                                len(multiprocessing.active_children()) - 1))
                        try:
                            input_queue_size = self.input_queue.qsize()
                        except Exception:
                            input_queue_size = -1
                        is_feeder_done = self.producer_thread_done_event.is_set()
                        self.log("...... Input Queue Status: %r. Is Feeder Done? %r" % (
                            input_queue_size, is_feeder_done))
                    if strikes > (self.child_failure_timeout * (1 + (scan_count / 500.0) * (
                            not self._has_remote_error))):
                        self.state = ProcessDispatcherState.running_local_workers_live
                        self.log(
                            ("...... Too much time has elapsed with"
                             " missing items (%d children still alive). Evaluating serially.") % (
                                 len(multiprocessing.active_children()) - 1,))
                        i += self._reconstruct_missing_work_items(
                            seen, hit_map, hit_to_scan, scan_hit_type_map)
                        has_work = False
                        self.debug("...... Processes")
                        for worker in self.workers:
                            self.debug("......... %r" % (worker,))
                        self.debug("...... IPC Manager: %r" % (self.ipc_manager,))
                continue

        consumer_end = time.time()
        self.debug("... Consumer Done (%0.3g sec.)" % (consumer_end - start_time))
        self.consumer_done_event.set()
        time.sleep(1) # Not a good solution but need to give workers a chance to sync
        i_spectrum_matches = sum(map(len, self.scan_solution_map.values()))
        self.log("... Finished Processing Matches (%d)" % (i_spectrum_matches,))
        self.clear_pool()
        self.debug("... Shutting Down Message Queue")
        self.log_controller.stop()
        self.debug("... Joining Feeder Thread (Done: %r)" % (self.producer_thread_done_event.is_set(), ))
        feeder_thread.join()
        dispatcher_end = time.time()
        self.log("... Dispatcher Finished (%0.3g sec.)" % (dispatcher_end - start_time))
        return self.scan_solution_map


class SpectrumIdentificationWorkerBase(Process, SpectrumEvaluatorBase):
    verbose = False

    def __init__(self, input_queue, output_queue, producer_done_event, consumer_done_event,
                 scorer_type, evaluation_args, spectrum_map, mass_shift_map, log_handler,
                 solution_packer):
        Process.__init__(self)
        if evaluation_args is None:
            evaluation_args = dict()
        self.daemon = True
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.producer_done_event = producer_done_event
        self.consumer_done_event = consumer_done_event
        self.scorer_type = scorer_type
        self.evaluation_args = evaluation_args

        self.solution_packer = solution_packer

        self.spectrum_map = spectrum_map
        self.mass_shift_map = mass_shift_map

        self.local_scan_map = LRUMapping(1000)
        self.local_mass_shift_map = dict({
            Unmodified.name: Unmodified
        })
        self.solution_map = dict()
        self._work_complete = Event()
        self._work_complete.clear()
        self.log_handler = log_handler
        self.token = uid()
        self.items_handled = 0
        self.result_buffer = []
        self.buffer_size = 1000
        self.last_sent_result = time.time()

    def log(self, message):
        """Send a normal logging message via :attr:`log_handler`

        Parameters
        ----------
        message : str
            The message to log
        """
        if self.log_handler is not None:
            self.log_handler(message)

    def debug(self, message):
        """Send a debugging message via :attr:`log_handler`

        Parameters
        ----------
        message : str
            The message to log
        """
        if self.verbose:
            self.log_handler("DEBUG::%s" % message)

    def fetch_scan(self, key):
        try:
            return self.local_scan_map[key]
        except KeyError:
            serialized_scan = self.spectrum_map[key]
            scan = pickle.loads(serialized_scan)
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
        """A helper method to encapsulate :attr:`_work_complete`'s ``is_set``
        method.

        Returns
        -------
        bool
        """
        return self._work_complete.is_set()

    def _append_to_result_buffer(self, payload):
        self.result_buffer.append(payload)
        if len(self.result_buffer) > self.buffer_size:
            self._flush_result_buffer()

    def _flush_result_buffer(self):
        if self.result_buffer:
            self.output_queue.put(self.result_buffer)
            self.result_buffer = []
            self.last_sent_result = time.time()

    def pack_output(self, target):
        """Transmit the completed identifications for a given target
        structure.

        Rather than returning these values directly, this implementation uses
        the output queue to send the target and its scores along an IPC queue.

        Parameters
        ----------
        target : object
            The structure being identified.
        """
        if self.solution_map:
            target_id = getattr(target, 'id', target)
            self._append_to_result_buffer((target_id, self.solution_map, self.token))
        self.solution_map = dict()

    def evaluate(self, scan, structure, evaluation_context=None, *args, **kwargs):
        raise NotImplementedError()

    def cleanup(self):
        """Send signals indicating the worker process is finished and do any
        final shared resource cleanup needed on the worker's side.

        This will set the :attr:`_work_complete` event and join :attr:`output_queue`
        """
        self.debug("... Process %s Setting Work Complete Flag. Processed %d structures" % (
            self.name, self.items_handled))
        try:
            self._work_complete.set()
        except (RemoteError, KeyError):
            self.log("An error occurred while cleaning up worker %r" % (self, ))
        self._flush_result_buffer()
        self.output_queue.put(SentinelToken(self.token))
        self.consumer_done_event.wait()
        # joining the queue may not be necessary if we depend upon consumer_event_done
        self.debug("... Process %s Queue Joining" % (self.name,))
        self.output_queue.join()
        self.debug("... Process %s Finished" % (self.name,))

    def before_task(self):
        '''A method to be overriden by subclasses that want to do something before
        starting the task loop.
        '''
        pass

    def task(self):
        """The worker process's main loop where it will poll for new work items,
        process incoming work items and send them back to the master process.
        """
        has_work = True
        self.items_handled = 0
        strikes = 0
        while has_work:
            try:
                payload = self.input_queue.get(True, 5)
                self.input_queue.task_done()
                strikes = 0
            except QueueEmptyException:
                if self.producer_done_event.is_set():
                    has_work = False
                    break
                else:
                    strikes += 1
                    self._flush_result_buffer()
                    if strikes % 1000 == 0:
                        self.log("... %d iterations without work for %r" % (strikes, self))
                    continue
            if self.items_handled % 100 == 0:
                if time.time() - self.last_sent_result > 60:
                    self._flush_result_buffer()
            # Handling a group of work items
            if isinstance(payload, dict):
                work_order = payload
                self.items_handled += 1
                try:
                    self.handle_group(work_order)
                except Exception:
                    message = "An error occurred while processing %r on %r:\n%s" % (
                        work_order, self, traceback.format_exc())
                    self.log(message)
                    break
            else: # Handling a single work item
                structure, scan_ids = payload
                self.items_handled += 1
                try:
                    self.handle_item(structure, scan_ids)
                except Exception:
                    message = "An error occurred while processing %r on %r:\n%s" % (
                        structure, self, traceback.format_exc())
                    self.log(message)
                    break
        self.cleanup()

    def run(self):
        new_name = getattr(self, 'process_name', None)
        if new_name is not None:
            TaskBase().try_set_process_name(new_name)
        try:
            self.before_task()
        except Exception:
            self.log("An exception occurred during before_task for %r.\n%s" % (
                self, traceback.format_exc()))
        try:
            self.task()
        except Exception:
            self.log("An exception occurred while executing %r.\n%s" % (
                self, traceback.format_exc()))
            self.cleanup()
