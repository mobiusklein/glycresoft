from csv import reader, writer
import os
import glob
import tempfile
import shutil

from glypy.utils import uid

from six import text_type, binary_type, PY3

from .spectrum_match import SpectrumMatch, SpectrumSolutionSet
from .ref import SpectrumReference, TargetReference


class TempFileManager(object):
    def __init__(self, base_directory=None):
        if base_directory is None:
            base_directory = tempfile.mkdtemp(prefix="glycresoft_tmp_")
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
    def __init__(self, sink, manager):
        self.manager = manager
        self.counter = 0
        self.sink = sink
        self._init_writer()

    def _init_writer(self):
        try:
            if PY3:
                self.handle = open(self.sink, 'wt', newline='')
            else:
                self.handle = open(self.sink, 'wb')
        except Exception:
            if hasattr(self.sink, 'write'):
                self.handle = self.sink
            else:
                raise
        self.writer = writer(self.handle)

    def write(self, solution_set):
        self.manager.put_scan(solution_set.scan)
        if PY3:
            values = [solution_set.scan.id]
            for spectrum_match in solution_set:
                self.manager.put_mass_shift(spectrum_match.mass_shift)
                for val in spectrum_match.pack():
                    if not isinstance(val, str):
                        val = str(val)
                    values.append(val)
            self.writer.writerow(values)
        else:
            values = [solution_set.scan.id]
            for spectrum_match in solution_set:
                self.manager.put_mass_shift(spectrum_match.mass_shift)
                values.extend(spectrum_match.pack())
            self.writer.writerow(list(map(bytes, values)))
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
    def __init__(self, source, manager, target_resolver=None, spectrum_match_type=SpectrumMatch):
        self.manager = manager
        self.source = source
        self.target_resolver = target_resolver
        self.spectrum_match_type = spectrum_match_type or SpectrumMatch
        self._init_reader()

    def _init_reader(self):
        try:
            if PY3:
                self.handle = open(self.source, 'rt')
            else:
                self.handle = open(self.source, 'rb')
        except Exception:
            if hasattr(self.source, 'read'):
                self.handle = self.source
            else:
                raise
        self.reader = reader(self.handle)

    def resolve_scan(self, scan_id):
        return self.manager.get_scan(scan_id)

    def resolve_target(self, target_id):
        if self.target_resolver is None:
            return self.manager.get_target(target_id)
        else:
            return self.target_resolver(target_id)

    def resolve_mass_shift(self, mass_shift_name):
        mass_shift = self.manager.get_mass_shift(mass_shift_name)
        return mass_shift

    def handle_line(self):
        row = next(self.reader)
        scan_id = str(row[0])
        spectrum_reference = self.resolve_scan(scan_id)
        i = 1
        n = len(row)
        members = []
        while i < n:
            match, i = self.spectrum_match_type.unpack(row, spectrum_reference, self, i)
            members.append(match)
        result = SpectrumSolutionSet(spectrum_reference, members)
        return result

    def __next__(self):
        return self.handle_line()

    next = __next__

    def __iter__(self):
        return self


class SpectrumSolutionResolver(object):
    def __init__(self):
        self.mass_shift_by_name = dict()
        self.target_cache = dict()
        self.scan_cache = dict()

    def clear(self):
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


class SpectrumMatchStore(SpectrumSolutionResolver):
    _reader_type = SpectrumSolutionSetReader
    _writer_type = SpectrumSolutionSetWriter

    def __init__(self, tempfile_manager=None):
        if tempfile_manager is None:
            tempfile_manager = TempFileManager()
        self.temp_manager = tempfile_manager
        super(SpectrumMatchStore, self).__init__()

    def clear(self):
        self.temp_manager.clear()
        super(SpectrumMatchStore, self).clear()

    def writer(self, key):
        return self._writer_type(self.temp_manager.get(key), self)

    def reader(self, key, target_resolver=None):
        return self._reader_type(self.temp_manager.get(key), self, target_resolver=target_resolver)


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
