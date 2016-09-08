from threading import Thread, Event

try:
    from Queue import Queue, Empty as QueueEmpty
except ImportError:
    from queue import Queue, Empty as QueueEmpty

from ms_deisotope.output import db


class ScanLoader(object):
    def __init__(self, sessiongetter, load_size=400):
        self.sessiongetter = sessiongetter
        self.queue = Queue(500)
        self.done_event = Event()
        self.load_size = load_size
        self._thread = Thread(target=self._worker_loop)

    def unspool_ordered(self):
        session = self.sessiongetter.session
        scans = session.query(db.MSScan).join(
            db.MSScan.precursor_information).order_by(
            db.PrecursorInformation.neutral_mass.desc()
        ).yield_per(self.load_size)
        return scans

    def _worker_loop(self):
        for scan in self.unspool_ordered():
            self.queue.put(scan.convert())
        self.done_event.set()
        print("No more items", self.is_done())

    def is_done(self):
        return self.done_event.is_set()

    def run(self):
        self._thread.start()

    def join(self):
        self._thread.join()


class ChunkSpooler(object):
    def __init__(self, sessiongetter, chunk_size=100):
        self.loader = ScanLoader(sessiongetter)
        self.loader.run()
        self.chunk_size = chunk_size

    def get_chunk(self):
        i = 0
        data = []
        if self.loader.is_done():
            return data
        has_more = True
        while i < self.chunk_size and has_more:
            if self.loader.is_done():
                has_more = False
                print "No more!"
                break
            try:
                item = self.loader.queue.get(timeout=3)
                data.append(item)
                i += 1
            except QueueEmpty:
                continue
        return data

    def __iter__(self):
        while True:
            bunch = self.get_chunk()
            if bunch:
                yield bunch
            else:
                print "Empty Bunch"
                if self.loader.is_done():
                    print "Loader Done"
                    self.loader.join()
                    print "Loader Joined"
                    raise StopIteration()
