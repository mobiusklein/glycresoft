'''Implements a multiprocessing deconvolution algorithm
'''
import os
import sys
import multiprocessing

from collections import deque

import ms_peak_picker
import ms_deisotope

import traceback

from ms_deisotope.processor import (
    ScanProcessor, MSFileLoader,
    NoIsotopicClustersError, EmptyScanError)

from ms_deisotope.feature_map.quick_index import index as build_scan_index

from ms_deisotope.data_source.common import ProcessedScan

import logging
from glycan_profiling.task import (
    TaskBase,
    log_handle,
    CallInterval)

from glycan_profiling.config import get_configuration


from multiprocessing import Process, JoinableQueue
try:
    from Queue import Empty as QueueEmpty
except ImportError:
    from queue import Empty as QueueEmpty


logger = logging.getLogger("glycan_profiler.preprocessor")


DONE = b"--NO-MORE--"
SCAN_STATUS_GOOD = b"good"
SCAN_STATUS_SKIP = b"skip"


user_config = get_configuration()
huge_tree = user_config.get("xml_huge_tree", False)

savgol = ms_peak_picker.scan_filter.SavitskyGolayFilter()
denoise = ms_peak_picker.scan_filter.FTICRBaselineRemoval(window_length=2.)


class ScanIDYieldingProcess(Process):

    def __init__(self, ms_file_path, queue, start_scan=None, max_scans=None, end_scan=None,
                 no_more_event=None, ignore_tandem_scans=False, batch_size=1, log_handler=None):
        Process.__init__(self)
        self.daemon = True
        self.ms_file_path = ms_file_path
        self.queue = queue
        self.loader = None

        self.start_scan = start_scan
        self.max_scans = max_scans
        self.end_scan = end_scan
        self.end_scan_index = None
        self.passed_first_batch = False
        self.ignore_tandem_scans = ignore_tandem_scans
        self.batch_size = batch_size

        self.no_more_event = no_more_event

    def log_handler(self, *message):
        log_handle.log(*message)

    def _make_scan_batch(self):
        batch = []
        scan_ids = []
        for _ in range(self.batch_size):
            try:
                bunch = next(self.loader)
                scan, products = bunch
                products = [
                    prod for prod in products if prod.index <= self.end_scan_index]
                if scan is not None:
                    scan_id = scan.id
                else:
                    scan_id = None
                product_scan_ids = [p.id for p in products]
            except StopIteration:
                break
            except Exception as e:
                self.log_handler("An error occurred in _make_scan_batch", e)
                break
            if not self.ignore_tandem_scans:
                batch.append((scan_id, product_scan_ids, True))
            else:
                batch.append((scan_id, product_scan_ids, False))
            scan_ids.append(scan_id)
        return batch, scan_ids

    def run(self):
        self.loader = MSFileLoader(self.ms_file_path, decode_binary=False)

        if self.start_scan is not None:
            try:
                self.loader.start_from_scan(
                    self.start_scan, require_ms1=self.loader.has_ms1_scans(), grouped=True)
            except IndexError as e:
                self.log_handler(
                    "An error occurred while locating start scan", e)
                self.loader.reset()
                self.loader.make_iterator(grouped=True)
            except AttributeError:
                self.log_handler(
                    "The reader does not support random access, start time will be ignored", e)
                self.loader.reset()
                self.loader.make_iterator(grouped=True)
        else:
            self.loader.make_iterator(grouped=True)

        count = 0
        last = 0
        if self.max_scans is None:
            max_scans = float('inf')
        else:
            max_scans = self.max_scans

        end_scan = self.end_scan
        if end_scan is None:
            try:
                self.end_scan_index = len(self.loader)
            except AttributeError:
                self.end_scan_index = sys.maxint
        else:
            self.end_scan_index = self.loader.get_scan_by_id(end_scan).index
        while count < max_scans:
            try:
                batch, ids = self._make_scan_batch()
                if len(batch) > 0:
                    self.queue.put(batch)
                count += len(ids)
                if (count - last) > 1000:
                    last = count
                    self.queue.join()
                if (end_scan in ids and end_scan is not None) or len(ids) == 0:
                    self.log_handler("End Scan Found")
                    break
            except StopIteration:
                break
            except Exception as e:
                self.log_handler("An error occurred while fetching scans", e)
                break

        if self.no_more_event is not None:
            self.no_more_event.set()
            self.log_handler(
                "All Scan IDs have been dealt. %d scan bunches." % (count,))
        else:
            self.queue.put(DONE)


class ScanBunchLoader(object):

    def __init__(self, mzml_loader):
        self.loader = mzml_loader
        self.queue = deque()

    def put(self, scan_id, product_scan_ids):
        self.queue.append((scan_id, product_scan_ids))

    def get(self):
        scan_id, product_scan_ids = self.queue.popleft()
        if scan_id is not None:
            precursor = self.loader.get_scan_by_id(scan_id)
        else:
            precursor = None
        products = [self.loader.get_scan_by_id(
            pid) for pid in product_scan_ids if pid is not None]
        if precursor:
            precursor.product_scans = products
        return (precursor, products)


class ScanTransformMixin(object):
    def log_error(self, error, scan_id, scan, product_scan_ids):
        tb = traceback.format_exc()
        self.log_handler(
            "An %r occurred for %s (index %r) in Process %r\n%s" % (
                error, scan_id, scan.index, multiprocessing.current_process(),
                tb))

    def _init_batch_store(self):
        self._batch_store = deque()

    def get_work(self, block=True, timeout=30):
        if self._batch_store:
            return self._batch_store.popleft()
        else:
            batch = self.input_queue.get(block, timeout)
            self._batch_store.extend(batch)
            result = self._batch_store.popleft()
            return result

    def log_message(self, message):
        self.log_handler(message + ", %r" %
                         (multiprocessing.current_process()))

    def skip_entry(self, index, ms_level):
        self.output_queue.put((SCAN_STATUS_SKIP, index, ms_level))

    def skip_scan(self, scan):
        self.output_queue.put((SCAN_STATUS_SKIP, scan.index, scan.ms_level))

    def send_scan(self, scan):
        scan = scan.pack()
        # this attribute is not needed, and for MS1 scans is dangerous
        # to pickle.
        # It can pull other scans which may not yet have been packed
        # into the message sent back to the main process which in
        # turn can form a reference cycle and eat a lot of memory
        scan.product_scans = []
        self.output_queue.put((scan, scan.index, scan.ms_level))

    def all_work_done(self):
        return self._work_complete.is_set()

    def make_scan_transformer(self, loader=None):
        raise NotImplementedError()


class ScanTransformingProcess(Process, ScanTransformMixin):
    """ScanTransformingProcess describes a child process that consumes scan id bunches
    from a shared input queue, retrieves the relevant scans, and preprocesses them using an
    instance of :class:`ms_deisotope.processor.ScanProcessor`, sending the reduced result
    to a shared output queue.

    Attributes
    ----------
    input_queue : multiprocessing.JoinableQueue
        A shared input queue which contains payloads of bunches of
        scan ids
    ms1_deconvolution_args : dict
        Parameters passed to :class:`ms_deisotope.processor.ScanProcessor`
    ms1_peak_picking_args : dict
        Parameters passed to :class:`ms_deisotope.processor.ScanProcessor`
    msn_deconvolution_args : dict
        Parameters passed to :class:`ms_deisotope.processor.ScanProcessor`
    msn_peak_picking_args : dict
        Parameters passed to :class:`ms_deisotope.processor.ScanProcessor`
    mzml_path : str
        Path to the spectral data file on disk
    no_more_event : multiprocessing.Event
        An event which will be set when the process feeding the input
        queue has run out of items to add, indicating that any QueueEmptyException
        should be treated as a signal to finish rather than to wait for
        new input
    output_queue : multiprocessing.JoinableQueue
        A shared output queue which this object will put
        :class:`ms_deisotope.data_source.common.ProcessedScan` bunches onto.
    """

    def __init__(self, mzml_path, input_queue, output_queue,
                 no_more_event=None, ms1_peak_picking_args=None,
                 msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None,
                 envelope_selector=None, ms1_averaging=0, log_handler=None,
                 deconvolute=True, verbose=False):
        if log_handler is None:
            def print_message(msg):
                print(msg)
            log_handler = print_message

        if ms1_peak_picking_args is None:
            ms1_peak_picking_args = {
                "transforms": [denoise, savgol],
                "start_mz": 250
            }
        if msn_peak_picking_args is None:
            msn_peak_picking_args = {
                "transforms": []
            }
        if ms1_deconvolution_args is None:
            ms1_deconvolution_args = {
                "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(35., 2),
                "charge_range": (1, 8),
                "averagine": ms_deisotope.glycopeptide
            }
        if msn_deconvolution_args is None:
            msn_deconvolution_args = {
                "scorer": ms_deisotope.scoring.MSDeconVFitter(10.),
                "charge_range": (1, 8),
                "averagine": ms_deisotope.glycopeptide
            }

        Process.__init__(self)
        self.verbose = verbose
        self._init_batch_store()
        self.daemon = True
        self.mzml_path = mzml_path
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.ms1_peak_picking_args = ms1_peak_picking_args
        self.msn_peak_picking_args = msn_peak_picking_args
        self.ms1_deconvolution_args = ms1_deconvolution_args
        self.msn_deconvolution_args = msn_deconvolution_args
        self.envelope_selector = envelope_selector
        self.ms1_averaging = ms1_averaging
        self.deconvolute = deconvolute

        self.transformer = None

        self.no_more_event = no_more_event
        self._work_complete = multiprocessing.Event()
        self.log_handler = log_handler

    def make_scan_transformer(self, loader=None):
        transformer = ScanProcessor(
            loader,
            ms1_peak_picking_args=self.ms1_peak_picking_args,
            msn_peak_picking_args=self.msn_peak_picking_args,
            ms1_deconvolution_args=self.ms1_deconvolution_args,
            msn_deconvolution_args=self.msn_deconvolution_args,
            loader_type=lambda x: x,
            envelope_selector=self.envelope_selector,
            ms1_averaging=self.ms1_averaging)
        return transformer

    def handle_scan_bunch(self, scan, product_scans, scan_id, product_scan_ids, process_msn=True):
        transformer = self.transformer
        # handle the MS1 scan if it is present
        if scan is not None:
            if len(scan.arrays[0]) == 0:
                self.skip_scan(scan)
            else:
                try:
                    scan, priorities, product_scans = transformer.process_scan_group(
                        scan, product_scans)
                    if scan is None:
                        # no way to report skip
                        pass
                    else:
                        if self.verbose:
                            self.log_message("Handling Precursor Scan %r with %d peaks" % (scan.id, len(scan.peak_set)))
                        if self.deconvolute:
                            transformer.deconvolute_precursor_scan(scan, priorities, product_scans)
                        self.send_scan(scan)
                except NoIsotopicClustersError as e:
                    self.log_message("No isotopic clusters were extracted from scan %s (%r)" % (
                        e.scan_id, len(scan.peak_set)))
                    self.skip_scan(scan)
                except EmptyScanError as e:
                    self.skip_scan(scan)
                except Exception as e:
                    self.skip_scan(scan)
                    self.log_error(e, scan_id, scan, (product_scan_ids))

        for product_scan in product_scans:
            # no way to report skip
            if product_scan is None:
                continue
            if len(product_scan.arrays[0]) == 0 or (not process_msn):
                self.skip_scan(product_scan)
                continue
            try:
                transformer.pick_product_scan_peaks(product_scan)
                if self.verbose:
                    self.log_message("Handling Product Scan %r with %d peaks (%0.3f/%0.3f, %r)" % (
                        product_scan.id, len(product_scan.peak_set), product_scan.precursor_information.mz,
                        product_scan.precursor_information.extracted_mz,
                        product_scan.precursor_information.defaulted))
                if self.deconvolute:
                    transformer.deconvolute_product_scan(product_scan)
                    if scan is None:
                        product_scan.precursor_information.default(orphan=True)
                self.send_scan(product_scan)
            except NoIsotopicClustersError as e:
                self.log_message("No isotopic clusters were extracted from scan %s (%r)" % (
                    e.scan_id, len(product_scan.peak_set)))
                self.skip_scan(product_scan)
            except EmptyScanError as e:
                self.skip_scan(product_scan)
            except Exception as e:
                self.skip_scan(product_scan)
                self.log_error(e, product_scan.id,
                               product_scan, (product_scan_ids))

    def run(self):
        loader = MSFileLoader(
            self.mzml_path, huge_tree=huge_tree, decode_binary=False)
        queued_loader = ScanBunchLoader(loader)

        has_input = True
        transformer = self.make_scan_transformer(loader)
        self.transformer = transformer

        nologs = ["deconvolution_scan_processor"]
        if not self.deconvolute:
            nologs.append("deconvolution")

        debug_mode = os.getenv("GLYCRESOFTDEBUG")
        if debug_mode:
            handler = logging.FileHandler("piped-deconvolution-debug-%s.log" % (os.getpid()), 'w')
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s:%(filename)s:%(lineno)-4d - %(levelname)s - %(message)s",
                "%H:%M:%S")
            handler.setFormatter(fmt)
        for logname in nologs:
            logger_to_silence = logging.getLogger(logname)
            if debug_mode:
                logger_to_silence.setLevel("DEBUG")
                logger_to_silence.addHandler(handler)
            else:
                logger_to_silence.propagate = False
                logger_to_silence.setLevel("CRITICAL")
                logger_to_silence.addHandler(logging.NullHandler())

        i = 0
        last = 0
        while has_input:
            try:
                scan_id, product_scan_ids, process_msn = self.get_work(True, 10)
                self.input_queue.task_done()
            except QueueEmpty:
                if self.no_more_event is not None and self.no_more_event.is_set():
                    has_input = False
                continue

            i += 1 + len(product_scan_ids)
            if scan_id == DONE:
                has_input = False
                break

            try:
                queued_loader.put(scan_id, product_scan_ids)
                scan, product_scans = queued_loader.get()
            except Exception as e:
                self.log_message("Something went wrong when loading bunch (%s): %r.\nRecovery is not possible." % (
                    (scan_id, product_scan_ids), e))

            self.handle_scan_bunch(scan, product_scans, scan_id, product_scan_ids, process_msn)
            if (i - last) > 1000:
                last = i
                self.output_queue.join()

        self.log_message("Done (%d scans)" % i)

        if self.no_more_event is None:
            self.output_queue.put((DONE, DONE, DONE))

        self._work_complete.set()


class ScanCollator(TaskBase):
    """Collates incoming scan bunches from multiple
    ScanTransformingProcesses, passing them along in
    the correct order.

    Attributes
    ----------
    count_jobs_done : int
        The number of scan bunches taken from :attr:`queue`
    count_since_last : int
        The number of work-cycles since the last scan bunch
        has been yielded
    done_event : multiprocessing.Event
        An IPC Event to indicate that all scan ids have been
        sent to the worker processes
    helper_producers : list
        A list of ScanTransformingProcesses
    include_fitted : bool
        Whether or not to save the raw fitted peaks for each
        scan produced. When this is `False`, they will be
        discarded and memory will be saved
    last_index : int
        The index of the last scan yielded through the iterator
        loop. This controls the next scan to be yielded and any
        waiting conditions
    primary_worker : ScanTransformingProcess
        The first worker to start consuming scans which will dictate
        the first handled index. Is required to run in isolation
        from other worker processes to insure that the first scan
        arrives in order
    queue : multiprocessing.Queue
        The IPC queue that all workers place their results on
        to be consumed and yielded in order
    started_helpers : bool
        Whether or not the additional workers in :attr:`helper_producers`
        have been started or not
    waiting : dict
        A mapping from scan index to `Scan` object. Used to serve
        scans through the iterator when their index is called for
    """
    _log_received_scans = False

    def __init__(self, queue, done_event, helper_producers=None, primary_worker=None,
                 include_fitted=False, input_queue=None):
        if helper_producers is None:
            helper_producers = []
        self.queue = queue
        self.last_index = None
        self.count_jobs_done = 0
        self.count_since_last = 0
        self.waiting = {}
        self.done_event = done_event
        self.helper_producers = helper_producers
        self.started_helpers = False
        self.primary_worker = primary_worker
        self.include_fitted = include_fitted
        self.input_queue = input_queue

    def all_workers_done(self):
        if self.done_event.is_set():
            if self.primary_worker.all_work_done():
                for helper in self.helper_producers:
                    if not helper.all_work_done():
                        return False
                return True
            else:
                return False
        return False

    def store_item(self, item, index):
        """Stores an incoming work-item for easy
        access by its `index` value. If configuration
        requires it, this will also reduce the number
        of peaks in `item`.

        Parameters
        ----------
        item : str or ProcessedScan
            Either a stub indicating why this work item
            is not
        index : int
            Scan index to store
        """
        if self._log_received_scans:
            self.log("-- received %d: %s" % (index, item))
        self.waiting[index] = item
        if not self.include_fitted and isinstance(item, ProcessedScan):
            item.peak_set = []

    def consume(self, timeout=10):
        """Fetches the next work item from the input
        queue :attr:`queue`, blocking for at most `timeout` seconds.

        Parameters
        ----------
        timeout : int, optional
            The duration to allow the process to block
            for while awaiting new work items.

        Returns
        -------
        bool
            Whether or not a new work item was found waiting
            on the :attr:`queue`
        """
        blocking = timeout != 0
        try:
            item, index, _ms_level = self.queue.get(blocking, timeout)
            self.queue.task_done()
            # DONE message may be sent many times.
            while item == DONE:
                item, index, _ms_level = self.queue.get(blocking, timeout)
                self.queue.task_done()
            self.store_item(item, index)
            return True
        except QueueEmpty:
            return False

    def start_helper_producers(self):
        """Starts the additional :class:`ScanTransformingProcess` workers
        in :attr:`helper_producers` if they have not been started already.

        Should only be invoked once
        """
        if self.started_helpers:
            return
        self.started_helpers = True
        for helper in self.helper_producers:
            if helper.is_alive():
                continue
            helper.start()

    def produce(self, scan):
        """Performs any final quality controls on the outgoing
        :class:`ProcessedScan` object and takes care of any internal
        details.

        Resets :attr:`count_since_last` to `0`.

        Parameters
        ----------
        scan : ProcessedScan
            The scan object being finalized for hand-off
            to client code

        Returns
        -------
        ProcessedScan
            The version of `scan` ready to be used by other
            parts of the program
        """
        self.count_since_last = 0
        return scan

    def count_pending_items(self):
        return len(self.waiting)

    def drain_queue(self):
        i = 0
        has_next = self.last_index + 1 not in self.waiting
        while (self.count_pending_items() < (1000 if has_next else 10)
               and self.consume(.1)):
            self.count_jobs_done += 1
            has_next = self.last_index + 1 not in self.waiting
            i += 1
        if i > 15:
            self.log("Drained Output Queue of %d Items" % (i, ))
        return i

    def print_state(self):
        try:
            if self.queue.qsize() > 0:
                self.log("%d since last work item" % (self.count_since_last,))
                keys = sorted(self.waiting.keys())
                if len(keys) > 5:
                    self.log("Waiting Keys: %r..." % (keys[:5],))
                else:
                    self.log("Waiting Keys: %r" % (keys,))
                self.log("%d Keys Total" % (len(self.waiting),))
                self.log("The last index handled: %r" % (self.last_index,))
                self.log("Number of items waiting in the queue: %d" %
                         (self.queue.qsize(),))
        except NotImplementedError:
            # Some platforms do not support qsize
            pass
        for worker in ([self.primary_worker] + list(self.helper_producers)):
            code = worker.exitcode
            if code is not None and code != 0:
                self.log("%r has exit code %r" % (worker, code))
                worker.join(5)

    def __iter__(self):
        has_more = True
        # Log the state of the collator every 3 minutes
        status_monitor = CallInterval(60 * 3, self.print_state)
        status_monitor.start()
        while has_more:
            if self.consume(1):
                self.count_jobs_done += 1
                try:
                    if self.queue.qsize() > 500:
                        self.drain_queue()
                except NotImplementedError:
                    # Some platforms do not support qsize. On these, always drain the queue.
                    self.drain_queue()
            if self.last_index is None:
                keys = sorted(self.waiting)
                if keys:
                    i = 0
                    n = len(keys)
                    found_content = False
                    while i < n:
                        scan = self.waiting.pop(keys[i])
                        if scan == SCAN_STATUS_SKIP:
                            self.last_index = keys[i]
                            i += 1
                            continue
                        else:
                            found_content = True
                            break
                    if found_content:
                        self.last_index = scan.index
                        yield self.produce(scan)
                    if self.last_index is not None:
                        self.start_helper_producers()
            elif self.last_index + 1 in self.waiting:
                while self.last_index + 1 in self.waiting:
                    scan = self.waiting.pop(self.last_index + 1)
                    if scan == SCAN_STATUS_SKIP:
                        self.last_index += 1
                        continue
                    else:
                        self.last_index = scan.index
                        yield self.produce(scan)
            elif len(self.waiting) == 0:
                if self.all_workers_done():
                    self.log("All Workers Claim Done.")
                    has_something = self.consume()
                    self.log("Checked Queue For Work: %r" % has_something)
                    if not has_something and len(self.waiting) == 0 and self.queue.empty():
                        has_more = False
            else:
                self.count_since_last += 1
                if self.count_since_last % 1000 == 0:
                    self.print_state()

        status_monitor.stop()


class ScanGeneratorBase(object):

    def configure_iteration(self, start_scan=None, end_scan=None, max_scans=None):
        raise NotImplementedError()

    def make_iterator(self, start_scan=None, end_scan=None, max_scans=None):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:  # pylint: disable=access-member-before-definition
            self._iterator = self.make_iterator()
        return next(self._iterator)

    def next(self):
        return self.__next__()

    def close(self):
        pass

    @property
    def scan_source(self):
        return None

    _deconvoluting = False

    @property
    def deconvoluting(self):
        return self._deconvoluting

    @deconvoluting.setter
    def deconvoluting(self, value):
        self._deconvoluting = value

    _ms1_averaging = 0

    @property
    def ms1_averaging(self):
        return self._ms1_averaging

    @ms1_averaging.setter
    def ms1_averaging(self, value):
        self._ms1_averaging = value

    _ignore_tandem_scans = False

    @property
    def ignore_tandem_scans(self):
        return self._ignore_tandem_scans

    @ignore_tandem_scans.setter
    def ignore_tandem_scans(self, value):
        self._ignore_tandem_scans = value

    _extract_only_tandem_envelopes = False

    @property
    def extract_only_tandem_envelopes(self):
        return self._extract_only_tandem_envelopes

    @extract_only_tandem_envelopes.setter
    def extract_only_tandem_envelopes(self, value):
        self._extract_only_tandem_envelopes = value


class ScanGenerator(TaskBase, ScanGeneratorBase):
    def __init__(self, ms_file, number_of_helpers=4,
                 ms1_peak_picking_args=None, msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None,
                 extract_only_tandem_envelopes=False, ignore_tandem_scans=False,
                 ms1_averaging=0, deconvolute=True):
        self.ms_file = ms_file
        self.time_cache = {}
        self.ignore_tandem_scans = ignore_tandem_scans

        self.scan_ids_exhausted_event = multiprocessing.Event()

        self._iterator = None

        self._scan_yielder_process = None
        self._deconv_process = None

        self._input_queue = None
        self._output_queue = None
        self._deconv_helpers = None
        self._order_manager = None

        self.number_of_helpers = number_of_helpers

        self.ms1_peak_picking_args = ms1_peak_picking_args
        self.msn_peak_picking_args = msn_peak_picking_args
        self.ms1_averaging = ms1_averaging

        self.deconvoluting = deconvolute
        self.ms1_deconvolution_args = ms1_deconvolution_args
        self.msn_deconvolution_args = msn_deconvolution_args
        self.extract_only_tandem_envelopes = extract_only_tandem_envelopes
        self._scan_interval_tree = None
        self.log_controller = self.ipc_logger()

    @property
    def scan_source(self):
        return self.ms_file

    def join(self):
        if self._scan_yielder_process is not None:
            self._scan_yielder_process.join()
        if self._deconv_process is not None:
            self._deconv_process.join()
        if self._deconv_helpers is not None:
            for helper in self._deconv_helpers:
                helper.join()

    def _terminate(self):
        if self._scan_yielder_process is not None:
            self._scan_yielder_process.terminate()
        if self._deconv_process is not None:
            self._deconv_process.terminate()
        if self._deconv_helpers is not None:
            for helper in self._deconv_helpers:
                helper.terminate()

    def _preindex_file(self):
        reader = MSFileLoader(self.ms_file, use_index=False, huge_tree=huge_tree)
        try:
            reader.prebuild_byte_offset_file(self.ms_file)
        except AttributeError:
            # the type does not support this type of indexing
            pass
        except IOError:
            # the file could not be written
            pass
        except Exception as e:
            # something else went wrong
            self.error("An error occurred while pre-indexing.", e)

    def _make_interval_tree(self, start_scan, end_scan):
        reader = MSFileLoader(self.ms_file, decode_binary=False)
        if start_scan is not None:
            start_ix = reader.get_scan_by_id(start_scan).index
        else:
            start_ix = 0
        if end_scan is not None:
            end_ix = reader.get_scan_by_id(end_scan).index
        else:
            end_ix = len(reader)
        reader.reset()
        _index, interval_tree = build_scan_index(
            reader, self.number_of_helpers + 1, (start_ix, end_ix))
        self._scan_interval_tree = interval_tree

    def _make_transforming_process(self):
        return ScanTransformingProcess(
            self.ms_file,
            self._input_queue,
            self._output_queue,
            self.scan_ids_exhausted_event,
            ms1_peak_picking_args=self.ms1_peak_picking_args,
            msn_peak_picking_args=self.msn_peak_picking_args,
            ms1_deconvolution_args=self.ms1_deconvolution_args,
            msn_deconvolution_args=self.msn_deconvolution_args,
            envelope_selector=self._scan_interval_tree,
            log_handler=self.log_controller.sender(),
            ms1_averaging=self.ms1_averaging,
            deconvolute=self.deconvoluting)

    def _make_collator(self):
        return ScanCollator(
            self._output_queue, self.scan_ids_exhausted_event, self._deconv_helpers,
            self._deconv_process, input_queue=self._input_queue,
            include_fitted=not self.deconvoluting)

    def _initialize_workers(self, start_scan=None, end_scan=None, max_scans=None):
        try:
            self._input_queue = JoinableQueue(int(1e6))
            self._output_queue = JoinableQueue(int(1e6))
        except OSError:
            # Not all platforms permit limiting the size of queues
            self._input_queue = JoinableQueue()
            self._output_queue = JoinableQueue()

        self._preindex_file()

        if self.extract_only_tandem_envelopes:
            self.log("Constructing Scan Interval Tree")
            self._make_interval_tree(start_scan, end_scan)

        self._terminate()
        self._scan_yielder_process = ScanIDYieldingProcess(
            self.ms_file, self._input_queue, start_scan=start_scan, end_scan=end_scan,
            max_scans=max_scans, no_more_event=self.scan_ids_exhausted_event,
            ignore_tandem_scans=self.ignore_tandem_scans, batch_size=1)
        self._scan_yielder_process.start()

        self._deconv_process = self._make_transforming_process()

        self._deconv_helpers = []

        for _i in range(self.number_of_helpers):
            self._deconv_helpers.append(self._make_transforming_process())
        self._deconv_process.start()

        self._order_manager = self._make_collator()

    def make_iterator(self, start_scan=None, end_scan=None, max_scans=None):
        self._initialize_workers(start_scan, end_scan, max_scans)

        for scan in self._order_manager:
            self.time_cache[scan.id] = scan.scan_time
            yield scan
        self.log_controller.stop()
        self.join()
        self._terminate()

    def configure_iteration(self, start_scan=None, end_scan=None, max_scans=None):
        self._iterator = self.make_iterator(start_scan, end_scan, max_scans)

    def convert_scan_id_to_retention_time(self, scan_id):
        return self.time_cache[scan_id]

    def close(self):
        self._terminate()
