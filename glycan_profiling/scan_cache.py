import os
import tempfile

import threading

import logging

try:
    from Queue import Queue, Empty as QueueEmptyException
except ImportError:
    from queue import Queue, Empty as QueueEmptyException

from ms_deisotope.data_source import MSFileLoader

from ms_deisotope.output.mzml import MzMLScanSerializer

from ms_deisotope.output.db import (
    BatchingDatabaseScanSerializer, ScanBunch,
    DatabaseScanDeserializer, FittedPeak,
    DeconvolutedPeak, DatabaseBoundOperation,
    MSScan)
from glycan_profiling.piped_deconvolve import ScanGeneratorBase

from .task import log_handle, TaskBase


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
        if self.current_precursor is not None:
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
            self.current_precursor = scan
        elif scan.ms_level == self.current_precursor.ms_level:
            self.save()
            self.current_precursor = scan
        else:
            self.current_products.append(scan)

    def commit(self):
        self.save()
        pass

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


class DatabaseScanCacheHandler(ScanCacheHandlerBase):
    commit_interval = 1000

    def __init__(self, connection, sample_name):
        self.serializer = BatchingDatabaseScanSerializer(connection, sample_name)
        self.commit_counter = 0
        self.last_commit_count = 0
        super(DatabaseScanCacheHandler, self).__init__()
        logger.info("Serializing scans under %r @ %r", self.serializer.sample_run, self.serializer.engine)

    def save_bunch(self, precursor, products):
        try:
            self.serializer.save(ScanBunch(precursor, products), commit=False)
            self.commit_counter += 1 + len(products)
            if self.commit_counter - self.last_commit_count > self.commit_interval:
                self.last_commit_count = self.commit_counter
                self.commit()
        except Exception as e:
            log_handle.error("An error occured while saving scans", e)

    def commit(self):
        super(DatabaseScanCacheHandler, self).save()
        self.serializer.commit()
        self.serializer.session.expunge_all()

    @classmethod
    def configure_storage(cls, path=None, name=None, source=None):
        if path is not None:
            if name is None:
                sample_name = os.path.basename(path)
            else:
                sample_name = name
            if not path.endswith(".db") and "://" not in path:
                path = os.path.splitext(path)[0] + '.db'
        elif path is None:
            path = tempfile.mkstemp()[1] + '.db'
            if name is None:
                sample_name = 'sample'
            else:
                sample_name = name
        return cls(path, sample_name)

    def complete(self):
        self.save()
        log_handle.log("Completing Serializer")
        self.serializer.complete()

    def _get_sample_run(self):
        return self.serializer.sample_run


class ThreadedDatabaseScanCacheHandler(DatabaseScanCacheHandler):
    def __init__(self, connection, sample_name):
        super(ThreadedDatabaseScanCacheHandler, self).__init__(connection, sample_name)
        self.queue = Queue(200)
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()
        self.log_inserts = False

    def _save_bunch(self, precursor, products):
        self.serializer.save(
            ScanBunch(precursor, products), commit=False)

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
                if next_bunch == DONE:
                    has_work = False
                    continue
                if self.log_inserts and (i % 100 == 0):
                    log_handle.log("Saving %r" % (next_bunch[0].id,))
                self._save_bunch(*next_bunch)
                self.commit_counter += 1 + len(next_bunch[1])
                i += 1

                if self.queue.qsize() > 0:
                    current_work = drain_queue()
                    for next_bunch in current_work:
                        if next_bunch == DONE:
                            has_work = False
                        else:
                            if self.log_inserts and (i % 100 == 0):
                                log_handle.log("Saving %r" % (next_bunch[0].id, ))
                            self._save_bunch(*next_bunch)
                            self.commit_counter += 1 + len(next_bunch[1])
                            i += 1

                if self.commit_counter - self.last_commit_count > self.commit_interval:
                    self.last_commit_count = self.commit_counter
                    log_handle.log("Syncing Scan Cache To Disk (%d items waiting)" % (self.queue.qsize(),))
                    self.serializer.commit()
                    if self.serializer.is_sqlite():
                        self.serializer.session.execute(
                            "PRAGMA wal_checkpoint(SQLITE_CHECKPOINT_RESTART);")
                    self.serializer.session.expunge_all()
            except QueueEmptyException:
                continue
            except Exception as e:
                log_handle.error("An error occurred while writing scans to disk", e)
        self.serializer.commit()
        self.serializer.session.expunge_all()

    def sync(self):
        self._end_thread()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _end_thread(self):
        logger.info(
            "Joining ThreadedDatabaseScanCacheHandler Worker. %d work items remaining",
            self.queue.qsize())
        self.log_inserts = True

        self.queue.put(DONE)
        if self.worker_thread is not None:
            self.worker_thread.join()

    def commit(self):
        super(DatabaseScanCacheHandler, self).save()
        self._end_thread()

    def complete(self):
        self.save()
        self._end_thread()
        super(ThreadedDatabaseScanCacheHandler, self).complete()


class MzMLScanCacheHandler(ScanCacheHandlerBase):
    def __init__(self, path, sample_name, n_spectra=None):
        if n_spectra is None:
            n_spectra = 2e5
        super(MzMLScanCacheHandler, self).__init__()
        self.path = path
        self.handle = open(path, 'wb')
        self.serializer = MzMLScanSerializer(self.handle, n_spectra, sample_name=sample_name)

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
            inst = cls(path, sample_name, n_spectra=n_spectra)
            try:
                description = reader.file_description()
                for key in description.get("fileContent", []):
                    inst.serializer.add_file_contents(key)
                for source_file in description.get('sourceFileList', []):
                    inst.add_source_file(source_file)
            except AttributeError:
                pass
            for trans in source.ms1_peak_picking_args.get("transforms"):
                inst.register_parameter("parameter: ms1-%s" % trans.__class__.__name__, repr(trans))
            if source.ms1_deconvolution_args.get("averagine"):
                inst.register_parameter(
                    "parameter: ms1-averagine", repr(source.ms1_deconvolution_args.get("averagine")))
            if source.ms1_deconvolution_args.get("scorer"):
                inst.register_parameter(
                    "parameter: ms1-scorer", repr(source.ms1_deconvolution_args.get("scorer")))
            if source.msn_peak_picking_args is not None:
                for trans in source.msn_peak_picking_args.get("transforms"):
                    inst.register_parameter("parameter: msn-%s" % trans.__class__.__name__, repr(trans))
            if source.msn_deconvolution_args.get("averagine"):
                inst.register_parameter(
                    "parameter: msn-averagine", repr(source.msn_deconvolution_args.get("averagine")))
            if source.msn_deconvolution_args.get("scorer"):
                inst.register_parameter(
                    "parameter: msn-scorer", repr(source.msn_deconvolution_args.get("scorer")))

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
        self.serializer.format()


class ThreadedMzMLScanCacheHandler(MzMLScanCacheHandler):
    def __init__(self, path, sample_name, n_spectra=None):
        super(ThreadedMzMLScanCacheHandler, self).__init__(path, sample_name, n_spectra)
        self.queue = Queue(200)
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _save_bunch(self, precursor, products):
        self.serializer.save(ScanBunch(precursor, products))

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
        try:
            super(ThreadedMzMLScanCacheHandler, self).complete()
        except Exception:
            import traceback
            traceback.print_exc()


class SampleRunDestroyer(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection, sample_run_id):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.sample_run_id = sample_run_id

    def delete_fitted_peaks(self):
        for ms_scan_id in self.session.query(MSScan.id).filter(MSScan.sample_run_id == self.sample_run_id):
            self.log("Clearing Fitted Peaks for %s" % ms_scan_id[0])
            self.session.query(FittedPeak).filter(FittedPeak.scan_id == ms_scan_id[0]).delete(
                synchronize_session=False)
            self.session.flush()

    def delete_deconvoluted_peaks(self):
        i = 0
        for ms_scan_id in self.session.query(MSScan.id).filter(MSScan.sample_run_id == self.sample_run_id):
            self.log("Clearing Deconvoluted Peaks for %s" % ms_scan_id[0])
            self.session.query(DeconvolutedPeak).filter(DeconvolutedPeak.scan_id == ms_scan_id[0]).delete(
                synchronize_session=False)
            i += 1
            if i % 100 == 0:
                self.session.flush()
        self.session.flush()

    def delete_ms_scans(self):
        self.session.query(MSScan).delete(synchronize_session=False)
        self.session.flush()
