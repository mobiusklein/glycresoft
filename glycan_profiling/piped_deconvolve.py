from collections import deque
import multiprocessing

import ms_peak_picker
import ms_deisotope

import traceback

from ms_deisotope.processor import (
    ScanProcessor, MSFileLoader, ScanIntervalTree,
    NoIsotopicClustersError)

from ms_deisotope.data_source.common import ProcessedScan

import logging
from .task import TaskBase, log_handle, CallInterval


from multiprocessing import Process, Queue
try:
    from Queue import Empty as QueueEmpty
except:
    from queue import Empty as QueueEmpty


logger = logging.getLogger("glycan_profiler.preprocessor")


DONE = b"--NO-MORE--"
SCAN_STATUS_GOOD = b"good"
SCAN_STATUS_SKIP = b"skip"

savgol = ms_peak_picker.scan_filter.SavitskyGolayFilter()
denoise = ms_peak_picker.scan_filter.FTICRBaselineRemoval(scale=2.)


class ScanIDYieldingProcess(Process):

    def __init__(self, mzml_path, queue, start_scan=None, max_scans=None, end_scan=None, no_more_event=None):
        Process.__init__(self)
        self.daemon = True
        self.mzml_path = mzml_path
        self.queue = queue
        self.loader = None

        self.start_scan = start_scan
        self.max_scans = max_scans
        self.end_scan = end_scan
        self.no_more_event = no_more_event

    def run(self):
        self.loader = MSFileLoader(self.mzml_path)

        index = 0
        if self.start_scan is not None:
            self.loader.start_from_scan(self.start_scan)

        count = 0
        if self.max_scans is None:
            max_scans = float('inf')
        else:
            max_scans = self.max_scans

        end_scan = self.end_scan

        while count < max_scans:
            try:
                scan, products = next(self.loader)
                scan_id = scan.id
                self.queue.put((scan_id, [p.id for p in products]))
                if scan_id == end_scan:
                    break
                index += 1
                count += 1
            except Exception as e:
                log_handle.error("An error occurred while fetching scans", e)
                break

        if self.no_more_event is not None:
            self.no_more_event.set()
            log_handle.log("All Scan IDs have been dealt. %r finished." % self)
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
        precursor = self.loader.get_scan_by_id(scan_id)
        products = [self.loader.get_scan_by_id(
            pid) for pid in product_scan_ids]
        precursor.product_scans = products
        return (precursor, products)

    def next(self):
        return self.get()

    def __next__(self):
        return self.get()


class ScanTransformingProcess(Process):
    """ScanTransformingProcess describes a child process that consumes scan id bunches
    from a shared input queue, retrieves the relevant scans, and preprocesses them using an
    instance of :class:`ms_deisotope.processor.ScanProcessor`, sending the reduced result
    to a shared output queue.

    Attributes
    ----------
    input_queue : multiprocessing.Queue
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
    output_queue : multiprocessing.Queue
        A shared output queue which this object will put
        :class:`ms_deisotope.data_source.common.ProcessedScan` bunches onto.
    """

    def __init__(self, mzml_path, input_queue, output_queue,
                 averagine=ms_deisotope.averagine.glycan, charge_range=(
                     -1, -8),
                 no_more_event=None, ms1_peak_picking_args=None, msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None,
                 envelope_selector=None, log_handler=None):
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
                "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(15., 2),
                "charge_range": charge_range,
                "averagine": averagine
            }
        if msn_deconvolution_args is None:
            msn_deconvolution_args = {
                "scorer": ms_deisotope.scoring.MSDeconVFitter(2.),
                "charge_range": charge_range,
                "averagine": averagine
            }

        Process.__init__(self)
        self.daemon = True
        self.mzml_path = mzml_path
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.averagine = dict(averagine)
        self.charge_range = charge_range

        self.ms1_peak_picking_args = ms1_peak_picking_args
        self.msn_peak_picking_args = msn_peak_picking_args
        self.ms1_deconvolution_args = ms1_deconvolution_args
        self.ms1_deconvolution_args.setdefault(
            "charge_range", self.charge_range)
        self.ms1_deconvolution_args.setdefault("averagine", averagine)
        self.msn_deconvolution_args = msn_deconvolution_args
        self.msn_deconvolution_args.setdefault(
            "charge_range", self.charge_range)
        self.msn_deconvolution_args.setdefault("averagine", averagine)
        self.envelope_selector = envelope_selector

        self.no_more_event = no_more_event
        self._work_complete = multiprocessing.Event()
        self.log_handler = log_handler

    def log_error(self, error, scan_id, scan, product_scan_ids):
        tb = traceback.format_exc()
        self.log_handler(
            "An %r occurred for %s (index %r) in Process %r\n%s" % (
                error, scan_id, scan.index, multiprocessing.current_process(),
                tb))

    def log_message(self, message):
        self.log_handler(message + ", %r" %
                         (multiprocessing.current_process()))

    def skip_scan(self, scan):
        self.output_queue.put((SCAN_STATUS_SKIP, scan.index, scan.ms_level))

    def send_scan(self, scan):
        self.output_queue.put((scan.pack(), scan.index, scan.ms_level))

    def all_work_done(self):
        return self._work_complete.is_set()

    def make_scan_transformer(self):
        transformer = ScanProcessor(
            None,
            ms1_peak_picking_args=self.ms1_peak_picking_args,
            msn_peak_picking_args=self.msn_peak_picking_args,
            ms1_deconvolution_args=self.ms1_deconvolution_args,
            msn_deconvolution_args=self.msn_deconvolution_args,
            loader_type=lambda x: x,
            envelope_selector=self.envelope_selector)
        return transformer

    def run(self):
        loader = MSFileLoader(self.mzml_path)
        queued_loader = ScanBunchLoader(loader)

        has_input = True
        transformer = self.make_scan_transformer()

        logger_to_silence = logging.getLogger("deconvolution_scan_processor")
        logger_to_silence.propagate = False
        logger_to_silence.addHandler(logging.NullHandler())

        i = 0
        while has_input:
            try:
                scan_id, product_scan_ids = self.input_queue.get(True, 10)
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

            if len(scan.arrays[0]) == 0:
                self.skip_scan(scan)
                continue

            try:
                scan, priorities, product_scans = transformer.process_scan_group(
                    scan, product_scans)
                transformer.deconvolute_precursor_scan(scan, priorities)
                self.send_scan(scan)
            except NoIsotopicClustersError as e:
                self.log_message("No isotopic clusters were extracted from scan %s (%r)" % (
                    e.scan_id, len(scan.peak_set)))
            except Exception as e:
                self.skip_scan(scan)
                self.log_error(e, scan_id, scan, (product_scan_ids))

            for product_scan in product_scans:
                if len(product_scan.arrays[0]) == 0:
                    self.skip_scan(product_scan)
                    continue
                try:
                    transformer.pick_product_scan_peaks(product_scan)
                    transformer.deconvolute_product_scan(product_scan)
                    self.send_scan(product_scan)
                except NoIsotopicClustersError as e:
                    self.log_message("No isotopic clusters were extracted from scan %s (%r)" % (
                        e.scan_id, len(product_scan.peak_set)))
                except Exception as e:
                    self.skip_scan(product_scan)
                    self.log_error(e, product_scan.id,
                                   product_scan, (product_scan_ids))

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

    def __init__(self, queue, done_event, helper_producers=None, primary_worker=None, include_fitted=False):
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
        index : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
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
            item, index, ms_level = self.queue.get(blocking, timeout)
            # DONE message may be sent many times.
            if item == DONE:
                item, index, ms_level = self.queue.get(blocking, timeout)
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
        while self.count_pending_items() < 500 and self.consume(0):
            self.count_jobs_done += 1
            i += 1
        return i

    def print_state(self):
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

    def __iter__(self):
        has_more = True
        # Log the state of the collator every 3 minutes
        status_monitor = CallInterval(60 * 3, self.print_state)
        status_monitor.start()
        while has_more:
            if self.consume(1):
                self.count_jobs_done += 1
                if self.queue.qsize() > 500:
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
                    if not has_something and len(self.waiting) == 0 and self.queue.qsize() == 0:
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
        if self._iterator is None:
            self._iterator = self.make_iterator()
        return next(self._iterator)

    def next(self):
        return self.__next__()

    def close(self):
        pass

    @property
    def scan_source(self):
        return None


class ScanGenerator(TaskBase, ScanGeneratorBase):

    def __init__(self, ms_file, averagine=ms_deisotope.averagine.glycan, charge_range=(-1, -8),
                 number_of_helper_deconvoluters=4, ms1_peak_picking_args=None, msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None,
                 extract_only_tandem_envelopes=False):
        self.ms_file = ms_file
        self.averagine = averagine
        self.time_cache = {}
        self.charge_range = charge_range

        self.scan_ids_exhausted_event = multiprocessing.Event()

        self._iterator = None

        self._picker_process = None
        self._deconv_process = None

        self._input_queue = None
        self._output_queue = None
        self._deconv_helpers = None
        self._order_manager = None

        self.number_of_helper_deconvoluters = number_of_helper_deconvoluters

        self.ms1_peak_picking_args = ms1_peak_picking_args
        self.msn_peak_picking_args = msn_peak_picking_args

        self.ms1_deconvolution_args = ms1_deconvolution_args
        self.msn_deconvolution_args = msn_deconvolution_args
        self.extract_only_tandem_envelopes = extract_only_tandem_envelopes
        self._scan_interval_tree = None
        self.log_controller = self.ipc_logger()

    @property
    def scan_source(self):
        return self.ms_file

    def join(self):
        if self._picker_process is not None:
            self._picker_process.join()
        if self._deconv_process is not None:
            self._deconv_process.join()
        if self._deconv_helpers is not None:
            for helper in self._deconv_helpers:
                helper.join()

    def _terminate(self):
        if self._picker_process is not None:
            self._picker_process.terminate()
        if self._deconv_process is not None:
            self._deconv_process.terminate()
        if self._deconv_helpers is not None:
            for helper in self._deconv_helpers:
                helper.terminate()

    def _make_interval_tree(self):
        reader = MSFileLoader(self.ms_file, use_index=False)
        self._scan_interval_tree = ScanIntervalTree.build(
            reader, time_radius=3.)

    def _make_transforming_process(self):
        return ScanTransformingProcess(
            self.ms_file,
            self._input_queue,
            self._output_queue,
            self.averagine,
            self.charge_range,
            self.scan_ids_exhausted_event,
            ms1_peak_picking_args=self.ms1_peak_picking_args,
            msn_peak_picking_args=self.msn_peak_picking_args,
            ms1_deconvolution_args=self.ms1_deconvolution_args,
            msn_deconvolution_args=self.msn_deconvolution_args,
            envelope_selector=self._scan_interval_tree,
            log_handler=self.log_controller.sender())

    def make_iterator(self, start_scan=None, end_scan=None, max_scans=None):
        self._input_queue = Queue(int(1e6))
        self._output_queue = Queue(5000)

        if self.extract_only_tandem_envelopes:
            self.log("Constructing Scan Interval Tree")
            self._make_interval_tree()

        self._terminate()

        self._picker_process = ScanIDYieldingProcess(
            self.ms_file, self._input_queue, start_scan=start_scan, end_scan=end_scan,
            max_scans=max_scans, no_more_event=self.scan_ids_exhausted_event)
        self._picker_process.start()

        self._deconv_process = self._make_transforming_process()

        self._deconv_helpers = []

        for i in range(self.number_of_helper_deconvoluters):
            self._deconv_helpers.append(self._make_transforming_process())
        self._deconv_process.start()

        self._order_manager = ScanCollator(
            self._output_queue, self.scan_ids_exhausted_event, self._deconv_helpers, self._deconv_process)

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
