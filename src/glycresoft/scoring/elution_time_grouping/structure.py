import csv

import io
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Mapping
from glycresoft.structure.structure_loader import GlycanCompositionDeltaCache

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from glycopeptidepy import PeptideSequence

from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenGlycanComposition

from glycresoft.structure import FragmentCachingGlycopeptide
from ms_deisotope.peak_dependency_network.intervals import SpanningMixin, IntervalTreeNode

from glycresoft.chromatogram_tree.utils import ArithmeticMapping
from glycresoft.chromatogram_tree.mass_shift import MassShiftBase, Unmodified
from glycresoft.chromatogram_tree.chromatogram import ChromatogramInterface

from glycresoft.tandem.spectrum_match import SpectrumMatchBase


def _try_parse(value):
    if isinstance(value, (int, float)):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value


def _get_apex_time(chromatogram):
    try:
        if hasattr(chromatogram, "as_arrays"):
            x, y = chromatogram.as_arrays()
        else:
            x, y = chromatogram.get_chromatogram().as_arrays()
        y = gaussian_filter1d(y, 1)
        return x[np.argmax(y)]
    except (AttributeError, TypeError):
        return chromatogram.apex_time


class ChromatogramProxy(object):

    weighted_neutral_mass: float
    apex_time: float
    total_signal: float
    glycan_composition: HashableGlycanComposition
    source: Any
    mass_shifts: List[MassShiftBase]
    weight: float
    kwargs: Dict[str, Any]

    def __init__(self, weighted_neutral_mass, apex_time, total_signal, glycan_composition, source=None,
                 mass_shifts=None, weight=1.0, **kwargs):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        self.weighted_neutral_mass = weighted_neutral_mass
        self.apex_time = apex_time
        self.total_signal = total_signal
        self.glycan_composition = glycan_composition
        self.source = source
        self.mass_shifts = mass_shifts
        self.weight = weight
        self.kwargs = kwargs

    @property
    def revised_from(self):
        return self.kwargs.get("revised_from")

    @revised_from.setter
    def revised_from(self, value):
        self.kwargs['revised_from'] = value

    @property
    def annotations(self):
        return self.kwargs

    @property
    def tag(self) -> Optional[int]:
        return self.kwargs.get('tag')

    @tag.setter
    def tag(self, value: int):
        self.kwargs['tag'] = value

    def __repr__(self):
        return "%s(%f, %f, %f, %s, %s)" % (
            self.__class__.__name__,
            self.weighted_neutral_mass, self.apex_time, self.total_signal,
            self.glycan_composition, self.kwargs)

    def pack(self):
        self.source = None

    @classmethod
    def from_obj(cls, obj, **kwargs):
        if isinstance(obj, ChromatogramInterface):
            return cls.from_chromatogram(obj)
        elif isinstance(obj, cls):
            return obj.__class__.from_chromatogram(obj, **kwargs)
        elif isinstance(obj, SpectrumMatchBase):
            return cls.from_spectrum_match(obj, **kwargs)
        else:
            return cls.from_chromatogram(obj)

    @classmethod
    def from_chromatogram(cls, obj, **kwargs):
        try:
            apex_time = _get_apex_time(obj)
        except (AttributeError, ValueError, TypeError):
            apex_time = obj.apex_time
        mass_shifts = getattr(obj, 'mass_shifts')
        kwargs.setdefault('mass_shifts', mass_shifts)
        inst = cls(
            obj.weighted_neutral_mass, apex_time, obj.total_signal,
            HashableGlycanComposition(obj.glycan_composition), obj, **kwargs)
        return inst

    @classmethod
    def from_spectrum_match(cls, spectrum_match, source=None, **kwargs):
        if source is None:
            source = spectrum_match
        return cls(
            spectrum_match.scan.precursor_information.neutral_mass, spectrum_match.scan.scan_time, 0.0,
            spectrum_match.target.glycan_composition, source, [spectrum_match.mass_shift], **kwargs)

    def get_chromatogram(self):
        return self.source.get_chromatogram()

    def _to_csv(self):
        d = self._prepare_state()
        d['mass_shifts'] = ';'.join(m.name for m in d['mass_shifts'])
        return d

    def _prepare_state(self):
        d = {
            "weighted_neutral_mass": self.weighted_neutral_mass,
            "apex_time": self.apex_time,
            "total_signal": self.total_signal,
            "glycan_composition": self.glycan_composition,
            "weight": float(self.weight),
            "mass_shifts": [m for m in self.mass_shifts],
        }
        d.update(self.kwargs)
        return d

    @classmethod
    def _from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def copy(self):
        dup = self._from_state(self._prepare_state())
        dup.source = self.source
        return dup

    def __eq__(self, other: 'ChromatogramProxy'):
        if other is None:
            return False
        if self.glycan_composition != other.glycan_composition:
            return False
        if not np.isclose(self.apex_time, other.apex_time):
            return False
        if not np.isclose(self.total_signal, other.total_signal):
            return False
        if self.mass_shifts != other.mass_shifts:
            return False
        if not np.isclose(self.weighted_neutral_mass, other.weighted_neutral_mass):
            return False
        return True

    def __ne__(self, other: 'ChromatogramProxy'):
        return not self == other

    def __hash__(self):
        return hash(self.glycan_composition)

    def __getstate__(self):
        state = self._prepare_state()
        state['glycan_composition'] = str(state['glycan_composition'])
        state['weight'] = self.weight
        return state

    def __setstate__(self, state):
        state = dict(state)
        self.glycan_composition = state.pop("glycan_composition", None)
        if self.glycan_composition is not None:
            if isinstance(self.glycan_composition, HashableGlycanComposition):
                pass
            else:
                self.glycan_composition = HashableGlycanComposition.parse(str(self.glycan_composition))
        self.apex_time = state.pop("apex_time", None)
        self.total_signal = state.pop("total_signal", None)
        self.weighted_neutral_mass = state.pop("weighted_neutral_mass", None)
        self.weight = state.pop('weight', 1.0)
        self.source = None
        self.mass_shifts = state.pop("mass_shifts", [Unmodified])
        self.kwargs = state

    def __getattr__(self, attr):
        try:
            return self.annotations[attr]
        except KeyError:
            raise AttributeError(attr)

    @classmethod
    def _csv_keys(cls, keys):
        return ['glycan_composition', 'apex_time', 'total_signal', 'weighted_neutral_mass', ] + \
            sorted(set(keys) - {'glycan_composition', 'apex_time',
                                'total_signal', 'weighted_neutral_mass', })

    @classmethod
    def to_csv(cls, instances: Iterable, fh: io.TextIOBase):
        cases = [c._to_csv() for c in instances]
        keys = cls._csv_keys(cases[0].keys())
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cases)

    @classmethod
    def _from_csv(cls, row):
        mass = float(row.pop("weighted_neutral_mass"))
        apex_time = float(row.pop("apex_time"))
        total_signal = float(row.pop("total_signal"))
        gc = HashableGlycanComposition.parse(row.pop("glycan_composition"))
        kwargs = {k: _try_parse(v) for k, v in row.items()}
        return cls(mass, apex_time, total_signal, gc, **kwargs)

    @classmethod
    def from_csv(cls, fh: io.TextIOBase):
        cases = []
        reader = csv.DictReader(fh)

        for row in reader:
            cases.append(cls._from_csv(row))
        return cases

    def update_glycan_composition(self, glycan_composition: HashableGlycanComposition) -> 'ChromatogramProxy':
        self.glycan_composition = glycan_composition

    def shift_glycan_composition(self, delta: HashableGlycanComposition):
        inst = self.copy()
        if isinstance(self.glycan_composition, HashableGlycanComposition):
            new_gc = self.glycan_composition + delta
        else:
            new_gc = HashableGlycanComposition(self.glycan_composition) + delta
        inst.update_glycan_composition(new_gc)
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
    def peptide_key(self) -> str:
        if "peptide_key" in self.kwargs:
            return self.kwargs["peptide_key"]
        structure = self.structure
        if isinstance(structure, str):
            peptide = str(PeptideSequence(structure).deglycosylate())
        else:
            peptide = str(structure.clone().deglycosylate())
        self.kwargs["peptide_key"] = peptide
        return peptide

    @property
    def structure(self) -> FragmentCachingGlycopeptide:
        if self._structure is None:
            self._structure = FragmentCachingGlycopeptide(
                str(self.kwargs["structure"]))
        return self._structure

    @structure.setter
    def structure(self, value: FragmentCachingGlycopeptide):
        self._structure = value
        self.kwargs['structure'] = str(value)
        if value and self.glycan_composition != value.glycan_composition:
            self.glycan_composition = HashableGlycanComposition(
                value.glycan_composition)

    @classmethod
    def from_chromatogram(cls, obj, **kwargs):
        gp = FragmentCachingGlycopeptide(str(obj.structure))
        result = super(GlycopeptideChromatogramProxy, cls).from_chromatogram(
            obj, structure=gp, **kwargs)
        _key = result.peptide_key
        return result

    @classmethod
    def from_spectrum_match(cls, spectrum_match, source=None, **kwargs):
        if source is None:
            source = spectrum_match
        gp = FragmentCachingGlycopeptide(str(spectrum_match.target))
        return cls(
            spectrum_match.scan.precursor_information.neutral_mass, spectrum_match.scan.scan_time, 0.0,
            spectrum_match.target.glycan_composition, source, [spectrum_match.mass_shift], structure=gp, **kwargs)

    def shift_glycan_composition(self, delta):
        inst = super(GlycopeptideChromatogramProxy,
                     self).shift_glycan_composition(delta)
        return inst

    def update_glycan_composition(self, glycan_composition: HashableGlycanComposition) -> 'GlycopeptideChromatogramProxy':
        original_structure = str(self.structure)
        super().update_glycan_composition(glycan_composition)
        structure = self.structure.clone()
        structure.glycan = self.glycan_composition
        self.structure = structure
        self.kwargs['original_structure'] = original_structure

    def __eq__(self, other: 'GlycopeptideChromatogramProxy'):
        result = super().__eq__(other)
        if not result:
            return result
        return self.structure == other.structure

    def __hash__(self):
        return hash(self.structure)


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


class GlycoformAggregator(Mapping):
    _interval_tree: IntervalTreeNode
    by_peptide: Dict[str, 'GlycoformGroup']
    factors: Dict[str, float]

    def __init__(self, glycoforms=None, *args, **kwargs):
        self.by_peptide = dict()
        self.factors = None
        self._interval_tree = None
        if glycoforms is not None:
            self.gather(glycoforms)

    def _get_peptide_key(self, chromatogram):
        if isinstance(chromatogram, GlycopeptideChromatogramProxy):
            return chromatogram.peptide_key
        return str(PeptideSequence(str(chromatogram.structure)).deglycosylate())

    def gather(self, iterable: Iterable[GlycopeptideChromatogramProxy]):
        for rec in iterable:
            key = self._get_peptide_key(rec)
            try:
                group = self.by_peptide[key]
            except KeyError:
                group = self.by_peptide[key] = GlycoformGroup([], key)
            group.append(rec)
        self._reindex()
        return self

    def _reindex(self):
        self.factors = self._infer_factors(self.glycoforms())
        self._interval_tree = IntervalTreeNode.build(self.values())
        return self

    def overlaps(self, start: float, end: float) -> List[GlycopeptideChromatogramProxy]:
        groups = self._interval_tree.overlaps(start, end)
        out = []
        for group in groups:
            out.extend(group.overlaps(start, end))
        return out

    @property
    def start_time(self):
        return min(v.start_time for v in self.values())

    @property
    def end_time(self):
        return max(v.end_time for v in self.values())

    def __getitem__(self, key) -> 'GlycoformGroup':
        if not isinstance(key, str):
            key = self._get_peptide_key(key)
        return self.by_peptide[key]

    def __setitem__(self, key, value: 'GlycoformGroup'):
        if not isinstance(key, str):
            key = self._get_peptide_key(key)
        self.by_peptide[key] = value

    def __contains__(self, key) -> bool:
        if not isinstance(key, str):
            key = self._get_peptide_key(key)
        return key in self.by_peptide

    def keys(self):
        return (self.by_peptide).keys()

    def __iter__(self):
        return self.glycoforms()

    def values(self):
        return self.by_peptide.values()

    def items(self):
        return self.by_peptide.items()

    def __len__(self):
        return len(self.by_peptide)

    def has_relative_pairs(self) -> bool:
        return any(len(v) > 1 for k, v in self.by_peptide.items())

    def _deltas_for(self, monosaccharide: str, include_pairs: bool = False):
        deltas = []
        pairs = []
        for _backbone, cases in self.by_peptide.items():
            for target in cases:
                gc = FrozenGlycanComposition.parse(
                    str(target.glycan_composition))
                gc[monosaccharide] += 1
                for case in cases:
                    if case.glycan_composition == gc:
                        deltas.append(case.apex_time - target.apex_time)
                        if include_pairs:
                            pairs.append((case, target, deltas[-1]))
        deltas = np.array(deltas)
        if include_pairs:
            return deltas, pairs
        return deltas

    def _infer_factors(self, chromatograms: Iterable[GlycopeptideChromatogramProxy]) -> List[str]:
        keys = set()
        for record in chromatograms:
            keys.update(record.glycan_composition)
        keys = sorted(map(str, keys))
        return keys

    def deltas(self, factors: Iterable[str] = None, include_pairs: bool = False):
        if factors is None:
            factors = self.factors
        return {f: self._deltas_for(f, include_pairs=include_pairs) for f in factors}

    def glycoforms(self) -> Iterator[GlycopeptideChromatogramProxy]:
        for group in self.values():
            for member in group:
                yield member

    def tag(self):
        i = 0
        for member in self:
            member.tag = i
            i += 1
        return self

    def get_tag(self, tag: int) -> GlycopeptideChromatogramProxy:
        for x in self:
            if x.tag == tag:
                return x
        raise KeyError(tag)

    def has_tag(self, tag: int) -> bool:
        for x in self:
            if x.tag == tag:
                return True
        return False


class GlycoformGroup(Sequence[GlycopeptideChromatogramProxy], SpanningMixin):
    members: List[GlycopeptideChromatogramProxy]
    group_id: str
    start: float
    end: float
    base_time: Optional[float]

    def __init__(self, members, group_id):
        self.members = list(members)
        self.group_id = group_id
        self.start_time = 0
        self.end_time = 0
        self.base_time = None

    @property
    def start_time(self):
        return self.start

    @start_time.setter
    def start_time(self, value):
        self.start = value

    @property
    def end_time(self):
        return self.end

    @end_time.setter
    def end_time(self, value):
        self.end = value

    def overlaps(self, start: float, end: float):
        return [m for m in self if start <= m.apex_time <= end]

    def _update(self):
        start_time = float('inf')
        end_time = 0.0
        for member in self.members:
            if member.apex_time < start_time:
                start_time = member.apex_time
            if member.apex_time > end_time:
                end_time = member.apex_time
        self.start_time = start_time
        self.end_time = end_time

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]

    def append(self, glycoform: GlycopeptideChromatogramProxy):
        self.members.append(glycoform)
        self._update()

    def extend(self, glycoforms: Iterable[GlycopeptideChromatogramProxy]):
        self.members.extend(glycoforms)
        self._update()

    def __repr__(self):
        return "{self.__class__.__name__}({self.members!r}, {self.group_id!r})".format(self=self)


class DeltaOverTimeFilter(object):
    def __init__(self, key, delta, time):
        self.key = key
        self.delta = np.array(delta)
        self.time = np.array(time)
        self.delta = self.gap_fill()
        self.smoothed = self.smooth()

    def smooth(self):
        return median_filter(self.delta, size=5, mode='nearest')

    def __array__(self):
        return self.smoothed

    def __getitem__(self, i):
        return self.smoothed[i]

    def __len__(self):
        return len(self.time)

    def gap_fill(self, width=2):
        x = self.delta.copy()
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = np.nanmean(np.concatenate(
                    (x[max(i - width, 0):i], x[i + 1:i + width])))
        return x

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.time, self.delta, label='Raw')
        ax.scatter(self.time, self.delta)
        ax.plot(self.time, self.smoothed,
                label="Smoothed", alpha=0.95, ls='--')
        return ax

    def search(self, time):
        lo = 0
        n = hi = len(self.time)
        while hi != lo:
            mid = (hi + lo) // 2
            x = self.time[mid]
            err = (x - time)
            if err == 0:
                return mid
            elif (hi - lo) == 1:
                err = abs(err)
                if mid != 0:
                    prev_err = self.time[mid - 1] - time
                    if abs(prev_err) < err:
                        return mid - 1
                if mid < n - 1:
                    next_err = self.time[mid + 1] - time
                    if abs(next_err) < err:
                        return mid + 1
                return mid
            elif err > 0:
                hi = mid
            else:
                lo = mid
