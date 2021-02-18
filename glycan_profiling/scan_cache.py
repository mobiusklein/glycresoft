import os
import threading

import logging

try:
    from Queue import Queue, Empty as QueueEmptyException
except ImportError:
    from queue import Queue, Empty as QueueEmptyException

from ms_deisotope.data_source import MSFileLoader, ScanBunch
from ms_deisotope.data_source.metadata.file_information import (
    SourceFile as MetadataSourceFile, FileInformation)

from ms_deisotope.output.mzml import MzMLScanSerializer


from glycan_profiling.task import log_handle


DONE = b'---NO-MORE---'

logger = logging.getLogger("glycan_profiler.scan_cache")


class ScanCacheHandlerBase(object):
    def __init__(self, *args, **kwargs):
        self.current_precursor = None
        self.current_products = []

    def reset(self):
        self.current_precursor = None
        self.current_products = []

    def save_bunch(self, precursor, products):
        raise NotImplementedError()

    def save(self):
        if self.current_precursor is not None or self.current_products:
            self.save_bunch(
                self.current_precursor, self.current_products)
            self.reset()

    def register_parameter(self, name, value):
        pass

    def complete(self):
        pass

    @classmethod
    def configure_storage(cls, path=None, name=None, source=None):
        return cls()

    def accumulate(self, scan):
        if self.current_precursor is None:
            if scan is not None:
                # If the scan is an MS1 scan, start accumulating a new bunch of scans
                if scan.ms_level == 1:
                    self.current_precursor = scan
                else:
                    # Otherwise this scan source may not have formal bunches of scans so we should
                    # just save the incoming scan.
                    self.current_products.append(scan)
                    self.save()
        elif scan.ms_level == self.current_precursor.ms_level:
            self.save()
            self.current_precursor = scan
        else:
            self.current_products.append(scan)

    def commit(self):
        self.save()

    def sync(self):
        self.commit()

    def _get_sample_run(self):
        return None

    @property
    def sample_run(self):
        return self._get_sample_run()


class NullScanCacheHandler(ScanCacheHandlerBase):
    def save_bunch(self, precursor, products):
        pass


class MzMLScanCacheHandler(ScanCacheHandlerBase):
    def __init__(self, path, sample_name, n_spectra=None, deconvoluted=True):
        if n_spectra is None:
            n_spectra = 2e5
        super(MzMLScanCacheHandler, self).__init__()
        self.path = path
        self.handle = open(path, 'wb')
        self.serializer = MzMLScanSerializer(
            self.handle, n_spectra, sample_name=sample_name,
            deconvoluted=deconvoluted)

    def _get_sample_run(self):
        return self.serializer.sample_run

    def register_parameter(self, name, value):
        self.serializer.add_processing_parameter(name, value)

    @classmethod
    def configure_storage(cls, path=None, name=None, source=None):
        if path is not None:
            if name is None:
                sample_name = os.path.basename(path)
            else:
                sample_name = name
        else:
            path = "processed.mzML"
        if source is not None:
            reader = MSFileLoader(source.scan_source)
            n_spectra = len(reader.index)
            deconvoluting = source.deconvoluting
            inst = cls(path, sample_name, n_spectra=n_spectra, deconvoluted=deconvoluting)
            try:
                description = reader.file_description()
            except AttributeError:
                description = FileInformation()
            source_file_metadata = MetadataSourceFile.from_path(source.scan_source)
            inst.serializer.add_file_information(description)
            try:
                inst.serializer.remove_file_contents("profile spectrum")
            except KeyError:
                pass
            inst.serializer.add_file_contents("centroid spectrum")
            if source_file_metadata not in description.source_files:
                inst.serializer.add_source_file(source_file_metadata)
            try:
                instrument_configs = reader.instrument_configuration()
                for config in instrument_configs:
                    inst.serializer.add_instrument_configuration(config)
            except Exception as e:
                log_handle.error(
                    "An error occurred while writing instrument configuration", e)
            for trans in source.ms1_peak_picking_args.get("transforms"):
                inst.register_parameter("parameter: ms1-%s" % trans.__class__.__name__, repr(trans))
            if deconvoluting:
                if source.ms1_deconvolution_args.get("averagine"):
                    inst.register_parameter(
                        "parameter: ms1-averagine", repr(source.ms1_deconvolution_args.get("averagine")))
                if source.ms1_deconvolution_args.get("scorer"):
                    inst.register_parameter(
                        "parameter: ms1-scorer", repr(source.ms1_deconvolution_args.get("scorer")))
                if source.ms1_averaging > 0:
                    inst.register_parameter("parameter: ms1-averaging", repr(source.ms1_averaging))
                if source.ignore_tandem_scans:
                    inst.register_parameter("parameter: ignore-tandem-scans", "")
                if source.extract_only_tandem_envelopes:
                    inst.register_parameter("parameter: extract-only-tandem-envelopes", "")

            if source.msn_peak_picking_args is not None:
                for trans in source.msn_peak_picking_args.get("transforms"):
                    inst.register_parameter("parameter: msn-%s" % trans.__class__.__name__, repr(trans))
            if deconvoluting:
                if source.msn_deconvolution_args.get("averagine"):
                    inst.register_parameter(
                        "parameter: msn-averagine", repr(source.msn_deconvolution_args.get("averagine")))
                if source.msn_deconvolution_args.get("scorer"):
                    inst.register_parameter(
                        "parameter: msn-scorer", repr(source.msn_deconvolution_args.get("scorer")))
            data_processing = inst.serializer.build_processing_method()
            inst.serializer.add_data_processing(data_processing)
        else:
            n_spectra = 2e5
            inst = cls(path, sample_name, n_spectra=n_spectra)
        # Force marshalling of controlled vocabularies early.
        inst.serializer.writer.param("32-bit float")
        return inst

    def save_bunch(self, precursor, products):
        self.serializer.save_scan_bunch(ScanBunch(precursor, products))

    def complete(self):
        self.save()
        self.serializer.complete()
        try:
            self.serializer.format()
        except OSError as e:
            if e.errno == 32:
                log_handle.log("Could not reformat the file in-place")
        except Exception:
            import traceback
            traceback.print_exc()


class ThreadedMzMLScanCacheHandler(MzMLScanCacheHandler):
    def __init__(self, path, sample_name, n_spectra=None, deconvoluted=True):
        super(ThreadedMzMLScanCacheHandler, self).__init__(
            path, sample_name, n_spectra, deconvoluted=deconvoluted)
        self.queue = Queue(200)
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _save_bunch(self, precursor, products):
        self.serializer.save(ScanBunch(precursor, products))
        try:
            precursor.clear()
            for product in products:
                product.clear()
        except AttributeError:
            pass

    def save_bunch(self, precursor, products):
        self.queue.put((precursor, products))

    def _worker_loop(self):
        has_work = True
        i = 0

        def drain_queue():
            current_work = []
            try:
                while len(current_work) < 300:
                    current_work.append(self.queue.get_nowait())
            except QueueEmptyException:
                pass
            if len(current_work) > 5:
                log_handle.log("Drained Write Queue of %d items" % (len(current_work),))
            return current_work

        while has_work:
            try:
                next_bunch = self.queue.get(True, 1)
                i += 1
                if next_bunch == DONE:
                    has_work = False
                    continue
                self._save_bunch(*next_bunch)
                if self.queue.qsize() > 0:
                    current_work = drain_queue()
                    for next_bunch in current_work:
                        i += 1
                        if next_bunch == DONE:
                            has_work = False
                        else:
                            self._save_bunch(*next_bunch)
                            i += 1
            except QueueEmptyException:
                continue
            except Exception as e:
                log_handle.error("An error occurred while writing scans to disk", e)

    def sync(self):
        self._end_thread()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _end_thread(self):
        self.queue.put(DONE)
        if self.worker_thread is not None:
            self.worker_thread.join()

    def commit(self):
        super(ThreadedMzMLScanCacheHandler, self).save()
        self._end_thread()

    def complete(self):
        self.save()
        self._end_thread()
        super(ThreadedMzMLScanCacheHandler, self).complete()
