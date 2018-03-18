from glypy.utils import uid

from csv import reader, writer
import os
import glob
import tempfile
import shutil

from .spectrum_matcher_base import SpectrumMatch, SpectrumSolutionSet
from .ref import SpectrumReference, TargetReference


class TempFileManager(object):
    def __init__(self, base_directory=None):
        if base_directory is None:
            base_directory = tempfile.mkdtemp("glycresoft"[::-1])
        else:
            try:
                os.makedirs(base_directory)
            except OSError:
                if os.path.exists(base_directory) and os.path.isdir(base_directory):
                    pass
                else:
                    raise
        self.base_directory = base_directory
        self.cache = {}

    def get(self, key=None):
        if key is None:
            _key = ""
        else:
            _key = key
        if key in self.cache:
            return self.cache[key]
        name = "%s_%x" % (_key, uid())
        path = os.path.join(self.base_directory, name)
        self.cache[key] = path
        return path

    def clear(self):
        shutil.rmtree(self.base_directory)

    def dir(self, pattern='*'):
        return glob.glob(os.path.join(self.base_directory, pattern))

    def __repr__(self):
        return "TempFileManager(%r)" % self.base_directory


class FileWrapperBase(object):
    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SpectrumSolutionSetWriter(FileWrapperBase):
    def __init__(self, path, manager):
        self.path = path
        self.handle = open(path, 'wb')
        self.writer = writer(self.handle)
        self.manager = manager
        self.counter = 0

    def write(self, solution_set):
        values = [solution_set.scan.id]
        self.manager.put_scan(solution_set.scan)
        for spectrum_match in solution_set:
            self.manager.put_mass_shift(spectrum_match.mass_shift)
            values.append(spectrum_match.target.id)
            values.append(spectrum_match.score)
            values.append(int(spectrum_match.best_match))
            values.append(spectrum_match.mass_shift.name)

        self.writer.writerow(list(map(str, values)))
        self.counter += 1

    def write_all(self, iterable):
        for item in iterable:
            self.write(item)
        self.flush()

    def extend(self, iterable):
        self.write_all(iterable)

    def append(self, solution_set):
        self.write(solution_set)

    def __len__(self):
        return self.counter


class SpectrumSolutionSetReader(FileWrapperBase):
    def __init__(self, path, manager, target_resolver=None):
        self.path = path
        self.manager = manager
        self.handle = open(path, 'rb')
        self.reader = reader(self.handle)
        self.target_resolver = target_resolver

    def resolve_scan(self, scan_id):
        return self.manager.get_scan(scan_id)

    def resolve_target(self, target_id):
        if self.target_resolver is None:
            return self.manager.get_target(target_id)
        else:
            return self.target_resolver(target_id)

    def handle_line(self):
        row = next(self.reader)
        scan_id = str(row[0])
        spectrum_reference = self.resolve_scan(scan_id)
        i = 1
        n = len(row)
        members = []
        while i < n:
            target_id = int(row[i])
            score = float(row[i + 1])
            try:
                best_match = bool(int(row[i + 2]))
            except ValueError:
                best_match = bool(row[i + 2])
            mass_shift_name = row[i + 3]
            mass_shift = self.manager.get_mass_shift(mass_shift_name)
            match = SpectrumMatch(
                spectrum_reference,
                self.resolve_target(target_id),
                score,
                best_match,
                mass_shift=mass_shift)
            members.append(match)
            i += 4
        result = SpectrumSolutionSet(spectrum_reference, members)
        return result

    def __next__(self):
        return self.handle_line()

    next = __next__

    def __iter__(self):
        return self


class SpectrumMatchStore(object):
    def __init__(self, tempfile_manager=None):
        if tempfile_manager is None:
            tempfile_manager = TempFileManager()
        self.temp_manager = tempfile_manager
        self.mass_shift_by_name = dict()
        self.target_cache = dict()
        self.scan_cache = dict()

    def clear(self):
        self.temp_manager.clear()
        self.scan_cache.clear()
        self.target_cache.clear()

    def put_scan(self, scan):
        self.scan_cache[scan.id] = scan

    def get_scan(self, scan_id):
        try:
            return self.scan_cache[scan_id]
        except KeyError:
            return SpectrumReference(scan_id, None)

    def put_mass_shift(self, mass_shift):
        self.mass_shift_by_name[mass_shift.name] = mass_shift

    def get_mass_shift(self, name):
        return self.mass_shift_by_name[name]

    def get_target(self, id):
        try:
            return self.target_cache[id]
        except KeyError:
            ref = TargetReference(id)
            self.target_cache[id] = ref
            return ref

    def writer(self, key):
        return SpectrumSolutionSetWriter(self.temp_manager.get(key), self)

    def reader(self, key, target_resolver=None):
        return SpectrumSolutionSetReader(self.temp_manager.get(key), self, target_resolver=target_resolver)


class FileBackedSpectrumMatchCollection(object):
    def __init__(self, store, key, target_resolver=None):
        self.store = store
        self.key = key
        self.target_resolver = target_resolver
        self._size = None

    def __iter__(self):
        return self.store.reader(self.key, self.target_resolver)

    def __len__(self):
        if self._size is not None:
            return self._size
        i = 0
        for row in self:
            i += 1
        self._size = i
        return self._size
