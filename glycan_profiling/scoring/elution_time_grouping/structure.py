import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d

from glycopeptidepy import PeptideSequence
from glypy.structure.glycan_composition import HashableGlycanComposition

from glycan_profiling.chromatogram_tree.utils import ArithmeticMapping
from glycan_profiling.chromatogram_tree.mass_shift import MassShift


def _get_apex_time(chromatogram):
    try:
        x, y = chromatogram.as_arrays()
        y = gaussian_filter1d(y, 1)
        return x[np.argmax(y)]
    except AttributeError:
        return chromatogram.apex_time


class ChromatogramProxy(object):
    def __init__(self, weighted_neutral_mass, apex_time, total_signal, glycan_composition, obj=None, mass_shifts=None, **kwargs):
        self.weighted_neutral_mass = weighted_neutral_mass
        self.apex_time = apex_time
        self.total_signal = total_signal
        self.glycan_composition = glycan_composition
        self.obj = obj
        self._mass_shifts = None
        self.kwargs = kwargs
        if mass_shifts:
            if isinstance(mass_shifts, str):
                mass_shifts = mass_shifts.split(";")
            self.kwargs['mass_shifts'] = ';'.join([getattr(m, 'name', m) for m in mass_shifts])

    @property
    def mass_shifts(self):
        if self._mass_shifts is None:
            self._mass_shifts = [
                MassShift(name, MassShift.get(name)) for name in self.annotations.get("mass_shifts", '').split(";")]
        return self._mass_shifts

    @property
    def annotations(self):
        return self.kwargs

    def __repr__(self):
        return "%s(%f, %f, %f, %s, %s)" % (
            self.__class__.__name__,
            self.weighted_neutral_mass, self.apex_time, self.total_signal,
            self.glycan_composition, self.kwargs)

    def pack(self):
        self.obj = None

    @classmethod
    def from_obj(cls, obj, **kwargs):
        try:
            chrom = obj.get_chromatogram()
            apex_time = _get_apex_time(chrom)
        except (AttributeError, ValueError, TypeError):
            apex_time = obj.apex_time
        mass_shifts = getattr(obj, 'mass_shifts')
        kwargs.setdefault('mass_shifts', mass_shifts)
        inst = cls(
            obj.weighted_neutral_mass, apex_time, obj.total_signal,
            obj.glycan_composition, obj, **kwargs)
        return inst

    def get_chromatogram(self):
        return self.obj.get_chromatogram()

    def _to_csv(self):
        d = {
            "weighted_neutral_mass": self.weighted_neutral_mass,
            "apex_time": self.apex_time,
            "total_signal": self.total_signal,
            "glycan_composition": self.glycan_composition,
        }
        d.update(self.kwargs)
        return d

    def copy(self):
        return self._from_csv(self._to_csv())

    def __getstate__(self):
        return self._to_csv()

    def __setstate__(self, state):
        state = dict(state)
        self.glycan_composition = state.pop("glycan_composition", None)
        self.apex_time = state.pop("apex_time", None)
        self.total_signal = state.pop("total_signal", None)
        self.weighted_neutral_mass = state.pop("weighted_neutral_mass", None)
        self.kwargs = state

    def __getattr__(self, attr):
        try:
            return self.annotations[attr]
        except KeyError:
            raise AttributeError(attr)

    @classmethod
    def _csv_keys(cls, keys):
        return ['glycan_composition', 'apex_time', 'total_signal', 'weighted_neutral_mass'] + \
            sorted(set(keys) - {'glycan_composition', 'apex_time',
                                'total_signal', 'weighted_neutral_mass'})

    @classmethod
    def to_csv(cls, instances, fh):
        cases = [c._to_csv() for c in instances]
        keys = cls._csv_keys(cases[0].keys())
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cases)

    @classmethod
    def _from_csv(cls, row):

        def _try_parse(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value

        mass = float(row.pop("weighted_neutral_mass"))
        apex_time = float(row.pop("apex_time"))
        total_signal = float(row.pop("total_signal"))
        gc = HashableGlycanComposition.parse(row.pop("glycan_composition"))
        kwargs = {k: _try_parse(v) for k, v in row.items()}
        return cls(mass, apex_time, total_signal, gc, **kwargs)

    @classmethod
    def from_csv(cls, fh):
        cases = []
        reader = csv.DictReader(fh)

        for row in reader:
            cases.append(cls._from_csv(row))
        return cases

    def shift_glycan_composition(self, delta):
        inst = self.__class__.from_obj(self)
        inst.glycan_composition = HashableGlycanComposition(self.glycan_composition) + delta
        return inst


class GlycopeptideChromatogramProxy(ChromatogramProxy):
    _structure = None

    @classmethod
    def _csv_keys(cls, keys):
        return ['structure', 'glycan_composition', 'apex_time',
                'total_signal', 'weighted_neutral_mass'] + \
            sorted(set(keys) - {'structure', 'glycan_composition',
                                'apex_time', 'total_signal', 'weighted_neutral_mass'})

    @property
    def structure(self):
        if self._structure is None:
            self._structure = PeptideSequence(str(self.kwargs["structure"]))
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value
        self.kwargs['structure'] = str(value)

    @classmethod
    def from_obj(cls, obj, **kwargs):
        gp = PeptideSequence(str(obj.structure))
        return super(GlycopeptideChromatogramProxy, cls).from_obj(obj, structure=gp, **kwargs)

    def shift_glycan_composition(self, delta):
        inst = super(GlycopeptideChromatogramProxy, self).shift_glycan_composition(delta)
        structure = self.structure.clone()
        structure.glycan = inst.glycan_composition
        inst.structure = structure
        inst.kwargs['original_structure'] = str(self.structure)
        return inst


class CommonGlycopeptide(object):
    def __init__(self, structure, mapping):
        self.mapping = mapping
        self.structure = structure

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __delitem__(self, key):
        del self.mapping[key]

    def keys(self):
        return self.mapping.keys()

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    @property
    def glycan_composition(self):
        return self.structure.glycan_composition

    @property
    def apex_times(self):
        return ArithmeticMapping({k: v.apex_time for k, v in self.mapping.items()})

    @property
    def total_signals(self):
        return ArithmeticMapping({k: v.total_signal for k, v in self.mapping.items()})

    def __repr__(self):
        template = "{self.__class__.__name__}({self.structure!s}, {count} observations)"
        count = len(self.mapping)
        return template.format(self=self, count=count)
