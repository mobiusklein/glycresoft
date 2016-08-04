import os
import tempfile

import threading

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

from ms_deisotope.output.db import DatabaseScanSerializer, ScanBunch, DatabaseScanDeserializer


DONE = b'---NO-MORE---'


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

    def configure_storage(cls, path=None, name=None):
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


class NullScanCacheHandler(ScanCacheHandlerBase):
    def save_bunch(self, precursor, products):
        pass


class DatabaseScanCacheHandler(ScanCacheHandlerBase):
    def __init__(self, connection, sample_name):
        self.serializer = DatabaseScanSerializer(connection, sample_name)
        super(DatabaseScanCacheHandler, self).__init__()

    def save_bunch(self, precursor, products):
        self.serializer.save(ScanBunch(precursor, products), commit=False)

    def commit(self):
        super(DatabaseScanCacheHandler, self).save()
        self.serializer.session.commit()

    @classmethod
    def configure_storage(cls, path=None, name=None):
        if path is not None:
            if name is None:
                sample_name = os.path.basename(path)
            else:
                sample_name = name
            if not path.endswith(".db"):
                path = os.path.splitext(path)[0] + '.db'
        elif path is None:
            path = tempfile.mkstemp()[1] + '.db'
            if name is None:
                sample_name = 'sample'
            else:
                sample_name = name
        return cls(path, sample_name)


class ThreadedDatabaseScanCacheHandler(DatabaseScanCacheHandler):
    def __init__(self, connection, sample_name):
        super(ThreadedDatabaseScanCacheHandler, self).__init__(connection, sample_name)
        self.queue = Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _save_bunch(self, precursor, products):
        self.serializer.save(
            ScanBunch(precursor, products), commit=False)

    def save_bunch(self, precursor, products):
        self.queue.put((precursor, products))

    def _worker_loop(self):
        has_work = True
        while has_work:
            try:
                next_bunch = self.queue.get(True, 10)
                if next_bunch == DONE:
                    has_work = False
                    continue
                self._save_bunch(*next_bunch)

            except Empty:
                continue
        self.serializer.session.commit()

    def sync(self):
        self._end_thread()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()

    def _end_thread(self):
        self.queue.put(DONE)
        if self.worker_thread is not None:
            self.worker_thread.join()

    def commit(self):
        super(DatabaseScanCacheHandler, self).save()
        self._end_thread()
