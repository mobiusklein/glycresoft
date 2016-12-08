from collections import deque
import multiprocessing
import threading

import ms_peak_picker
import ms_deisotope

import traceback

from ms_deisotope.processor import MzMLLoader, ScanProcessor

import logging
from .task import TaskBase, log_handle


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


class CallInterval(object):
    """Call a function every `interval` seconds from
    a separate thread.

    Attributes
    ----------
    stopped: threading.Event
        A semaphore lock that controls when to run `call_target`
    call_target: callable
        The thing to call every `interval` seconds
    args: iterable
        Arguments for `call_target`
    interval: number
        Time between calls to `call_target`
    """

    def __init__(self, interval, call_target, *args):
        self.stopped = threading.Event()
        self.interval = interval
        self.call_target = call_target
        self.args = args
        self.thread = threading.Thread(target=self.mainloop)
        self.thread.daemon = True

    def mainloop(self):
        while not self.stopped.wait(self.interval):
            try:
                self.call_target(*self.args)
            except Exception, e:
                logger.exception("An error occurred in %r", self, exc_info=e)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()


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
        self.loader = MzMLLoader(self.mzml_path)

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
                if scan_id == end_scan:
                    break
                self.queue.put((scan_id, [p.id for p in products]))
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
        products = [self.loader.get_scan_by_id(pid) for pid in product_scan_ids]
        precursor.product_scans = products
        return (precursor, products)

    def next(self):
        return self.get()

    def __next__(self):
        return self.get()


class ScanTransformingProcess(Process):
    def __init__(self, mzml_path, input_queue, output_queue,
                 averagine=ms_deisotope.averagine.glycan, charge_range=(-1, -8),
                 no_more_event=None, ms1_peak_picking_args=None, msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None):

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
                "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(15, 2.),
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
        self.ms1_deconvolution_args.setdefault("charge_range", self.charge_range)
        self.ms1_deconvolution_args.setdefault("averagine", averagine)
        self.msn_deconvolution_args = msn_deconvolution_args
        self.msn_deconvolution_args.setdefault("charge_range", self.charge_range)
        self.msn_deconvolution_args.setdefault("averagine", averagine)

        self.no_more_event = no_more_event
        self._work_complete = multiprocessing.Event()

    def log_error(self, error, scan_id, scan, product_scan_ids):
        traceback.print_exc()
        logger.exception(
            "An error occurred for %s (index %r) in Process %r",
            scan_id, scan.index, multiprocessing.current_process(),
            exc_info=error)

    def log_message(self, message, *args):
        logger.info(message + "%r, %r" % (args, multiprocessing.current_process()))

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
            loader_type=lambda x: x)
        return transformer

    def run(self):
        loader = MzMLLoader(self.mzml_path)
        queued_loader = ScanBunchLoader(loader)

        has_input = True
        transformer = self.make_scan_transformer()

        logger_to_silence = logging.getLogger("deconvolution_scan_processor")
        logger_to_silence.propagate = False
        logger_to_silence.addHandler(logging.NullHandler())

        while has_input:
            try:
                scan_id, product_scan_ids = self.input_queue.get(True, 20)
            except QueueEmpty:
                if self.no_more_event is not None and self.no_more_event.is_set():
                    has_input = False
                continue

            # print("Handling Scan %r" % scan_id)
            if scan_id == 'scanId=4500801':
                print("Handling target scan", scan_id, self)

            if scan_id == DONE:
                has_input = False
                break

            queued_loader.put(scan_id, product_scan_ids)
            scan, product_scans = queued_loader.get()

            if len(scan.arrays[0]) == 0:
                self.skip_scan(scan)
                continue

            try:
                scan, priorities, product_scans = transformer.process_scan_group(scan, product_scans)
                transformer.deconvolute_precursor_scan(scan, priorities)
                self.send_scan(scan)
                if scan_id == 'scanId=4500801':
                    print("Sent target scan", scan, scan.index, self)
            except Exception as e:
                if scan_id == 'scanId=4500801':
                    print("Error on target scan", scan, scan.index, e)
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
                except Exception as e:
                    self.skip_scan(product_scan)
                    self.log_error(e, product_scan.id, product_scan, (product_scan_ids))

        self.log_message("Done")

        if self.no_more_event is None:
            self.output_queue.put((DONE, DONE, DONE))

        self._work_complete.set()


class ScanCollator(TaskBase):
    def __init__(self, queue, done_event, helper_producers=None, primary_worker=None):
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

    def consume(self, timeout=10):
        try:
            item, index, status = self.queue.get(True, timeout)
            if item == DONE:
                item, index, status = self.queue.get(True, timeout)
            self.waiting[index] = item
            return True
        except QueueEmpty:
            return False

    def start_helper_producers(self):
        if self.started_helpers:
            return
        self.started_helpers = True
        for helper in self.helper_producers:
            if helper.is_alive():
                continue
            helper.start()

    def produce(self, scan):
        self.count_since_last = 0
        return scan

    def print_state(self):
        if self.queue.qsize() > 0:
            self.log("%d since last work item" % (self.count_since_last,))
            keys = sorted(self.waiting.keys())
            if len(keys) > 20:
                self.log("Waiting Keys: %r..." % (keys[:21],))
            else:
                self.log("Waiting Keys: %r" % (keys,))
            self.log("%d Keys Total" % (len(self.waiting),))
            self.log("The last index handled: %r" % (self.last_index,))
            self.log("Number of items waiting in the queue: %d" % (self.queue.qsize(),))

    def __iter__(self):
        has_more = True
        # Log the state of the collator every 3 minutes
        status_monitor = CallInterval(60 * 3, self.print_state)
        status_monitor.start()
        while has_more:
            if self.consume(1):
                self.count_jobs_done += 1
            if self.last_index is None:
                keys = sorted(self.waiting)
                if keys:
                    scan = self.waiting.pop(keys[0])
                    if scan == SCAN_STATUS_SKIP:
                        self.log("Scan Skipped")
                        self.last_index = keys[0]
                        continue
                    self.last_index = scan.index
                    yield self.produce(scan)
                    self.start_helper_producers()
            elif self.last_index + 1 in self.waiting:
                scan = self.waiting.pop(self.last_index + 1)
                if scan == SCAN_STATUS_SKIP:
                    self.last_index += 1
                    continue
                self.last_index = scan.index
                yield self.produce(scan)
            elif len(self.waiting) == 0:
                if self.all_workers_done():
                    self.log("All Workers Claim Done.")
                    has_something = self.consume()
                    self.log("Checked Queue For Work: %r" % has_something)
                    if not has_something:
                        has_more = False
            else:
                self.count_since_last += 1
                if self.count_since_last % 25 == 0:
                    self.print_state()
                    # if self.all_workers_done():
                    #     if self.queue.qsize() == 0:
                    #         self.log("Scan Missing and All Workers Claim Done and Queue Empty. Skipping Index")
                    #         self.last_index += 1
                    #     else:
                    #         self.log("Scan Missing and All Workers Claim Done. Draining Queue.")
                    #         self.consume(1)

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


class ScanGenerator(ScanGeneratorBase):

    def __init__(self, mzml_file, averagine=ms_deisotope.averagine.glycan, charge_range=(-1, -8),
                 number_of_helper_deconvoluters=4, ms1_peak_picking_args=None, msn_peak_picking_args=None,
                 ms1_deconvolution_args=None, msn_deconvolution_args=None):
        self.mzml_file = mzml_file
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

    @property
    def scan_source(self):
        return self.mzml_file

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

    def _make_transforming_process(self):
        return ScanTransformingProcess(
            self.mzml_file,
            self._input_queue, self._output_queue, self.averagine, self.charge_range, self.scan_ids_exhausted_event,
            ms1_peak_picking_args=self.ms1_peak_picking_args, msn_peak_picking_args=self.msn_peak_picking_args,
            ms1_deconvolution_args=self.ms1_deconvolution_args, msn_deconvolution_args=self.msn_deconvolution_args)

    def make_iterator(self, start_scan=None, end_scan=None, max_scans=None):
        self._input_queue = Queue(int(1e6))
        self._output_queue = Queue(1000)

        self._terminate()

        self._picker_process = ScanIDYieldingProcess(
            self.mzml_file, self._input_queue, start_scan=start_scan, end_scan=end_scan,
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
        self.join()
        self._terminate()

    def configure_iteration(self, start_scan=None, end_scan=None, max_scans=None):
        self._iterator = self.make_iterator(start_scan, end_scan, max_scans)

    def convert_scan_id_to_retention_time(self, scan_id):
        return self.time_cache[scan_id]

    def close(self):
        self._terminate()
