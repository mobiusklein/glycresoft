from glypy.utils import uid

from csv import reader, writer
import os
import tempfile
import shutil

from .spectrum_matcher_base import SpectrumMatch
from .ref import SpectrumReference, TargetReference


class TempFileManager(object):
    def __init__(self, base_directory=None):
        if base_directory is None:
            base_directory = tempfile.mkdtemp("glycresoft"[::-1])
        else:
            os.makedirs(base_directory)
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

    def __repr__(self):
        return "TempFileManager(%s)" % self.base_directory


class SpectrumMatchWriter(object):
    def __init__(self, path, manager):
        self.path = path
        self.handle = open(path, 'wb')
        self.writer = writer(self.handle)
        self.manager = manager

    def write(self, spectrum_match):
        self.manager.put_mass_shift(spectrum_match.mass_shift)
        self.writer.writerow(list(map(str, [
            spectrum_match.scan.id,
            spectrum_match.target.id,
            spectrum_match.score,
            spectrum_match.best_match,
            spectrum_match.mass_shift.name
        ])))

    def write_all(self, iterable):
        for item in iterable:
            self.write(item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()


class SpectrumMatchReader(object):
    def __init__(self, path, manager):
        self.path = path
        self.handle = open(path, 'rb')
        self.reader = reader(self.handle)
        self.manager = manager

    def handle_line(self):
        row = next(self.reader)
        scan_id = str(row[0])
        target_id = int(row[1])
        score = float(row[2])
        best_match = bool(row[3])
        mass_shift_name = row[4]
        mass_shift = self.manager.get_mass_shift(mass_shift_name)
        return SpectrumMatch(
            SpectrumReference(scan_id, None),
            TargetReference(target_id),
            score,
            best_match,
            mass_shift=mass_shift)

    def __next__(self):
        return self.handle_line()

    next = __next__

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()


class SpectrumMatchStore(object):
    def __init__(self, base_directory=None):
        self.temp_manager = TempFileManager(base_directory)
        self.mass_shift_by_name = dict()

    def put_mass_shift(self, mass_shift):
        self.mass_shift_by_name[mass_shift.name] = mass_shift

    def get_mass_shift(self, name):
        return self.mass_shift_by_name[name]

    def writer(self, key):
        return SpectrumMatchWriter(self.temp_manager.get(key), self)

    def reader(self, key):
        return SpectrumMatchReader(self.temp_manager.get(key), self)
