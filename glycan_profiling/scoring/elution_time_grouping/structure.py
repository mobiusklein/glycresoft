import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d

from glycopeptidepy import PeptideSequence
from glypy.structure.glycan_composition import HashableGlycanComposition

from glycan_profiling.chromatogram_tree.utils import ArithmeticMapping


def _get_apex_time(chromatogram):
    try:
        x, y = chromatogram.as_arrays()
        y = gaussian_filter1d(y, 1)
        return x[np.argmax(y)]
    except AttributeError:
        return chromatogram.apex_time


class ChromatogramProxy(object):
    def __init__(self, weighted_neutral_mass, apex_time, total_signal, glycan_composition, obj=None, **kwargs):
        self.weighted_neutral_mass = weighted_neutral_mass
        self.apex_time = apex_time
        self.total_signal = total_signal
        self.glycan_composition = glycan_composition
        self.obj = obj
        self.kwargs = kwargs

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

    @classmethod
    def to_csv(cls, instances, fh):
        cases = [c._to_csv() for c in instances]
        keys = list(cases[0].keys())
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cases)

    @classmethod
    def from_csv(cls, fh):
        cases = []
        reader = csv.DictReader(fh)

        def _try_parse(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value

        for row in reader:
            mass = float(row.pop("weighted_neutral_mass"))
            apex_time = float(row.pop("apex_time"))
            total_signal = float(row.pop("total_signal"))
            gc = HashableGlycanComposition.parse(row.pop("glycan_composition"))
            kwargs = {k: _try_parse(v) for k, v in row.items()}
            cases.append(cls(mass, apex_time, total_signal, gc, **kwargs))
        return cases

    def shift_glycan_composition(self, delta):
        inst = self.__class__.from_obj(self)
        inst.glycan_composition = HashableGlycanComposition(self.glycan_composition) - delta
        return inst


class GlycopeptideChromatogramProxy(ChromatogramProxy):
    _structure = None

    @property
    def structure(self):
        if self._structure is None:
            self._structure = PeptideSequence(str(self.kwargs["structure"]))
        return self._structure

    @structure.setter
    def structure_setter(self, value):
        self._structure = value

    @classmethod
    def from_obj(cls, obj, **kwargs):
        gp = PeptideSequence(str(obj.structure))
        return super(GlycopeptideChromatogramProxy, cls).from_obj(obj, structure=gp, **kwargs)


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
        count = len(self.structure)
        return template.format(self=self, count=count)
