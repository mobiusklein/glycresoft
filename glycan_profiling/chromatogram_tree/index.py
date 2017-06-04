from collections import defaultdict

from . import (smooth_overlaps, build_rt_interval_tree)


def binary_search_with_flag(array, mass, error_tolerance=1e-5):
    """Binary search an ordered array of objects with :attr:`neutral_mass`
    using a PPM error tolerance of `error_toler

    Parameters
    ----------
    array : list
        An list of objects, sorted over :attr:`neutral_mass` in increasing order
    mass : float
        The mass to search for
    error_tolerance : float, optional
        The PPM error tolerance to use when deciding whether a match has been found

    Returns
    -------
    int:
        The index in `array` of the best match
    bool:
        Whether or not a match was actually found, used to
        signal behavior to the caller.
    """
    lo = 0
    n = hi = len(array)
    while hi != lo:
        mid = (hi + lo) // 2
        x = array[mid]
        err = (x.neutral_mass - mass) / mass
        if abs(err) <= error_tolerance:
            best_index = mid
            best_error = err
            i = mid - 1
            while i >= 0:
                x = array[i]
                err = abs((x.neutral_mass - mass) / mass)
                if err < best_error:
                    best_error = err
                    best_index = i
                i -= 1

            i = mid + 1
            while i < n:
                x = array[i]
                err = abs((x.neutral_mass - mass) / mass)
                if err < best_error:
                    best_error = err
                    best_index = i
                i += 1
            return best_index, True
        elif (hi - lo) == 1:
            return mid, False
        elif err > 0:
            hi = mid
        elif err < 0:
            lo = mid
    return 0, False


class ChromatogramFilter(object):
    """An ordered collection of Chromatogram-like objects with fast searching
    and filtering features. Supports Sequence operations.

    Attributes
    ----------
    chromatograms: list of Chromatogram
        list of chromatogram-like objects, ordered by neutral mass
    key_map: dict
        A mapping between values appearing in the :attr:`Chromatogram.key` and :class:`DisjointChromatogramSet`
        instances containing all occurrences of that key, ordered by time.
    rt_interval_tree: IntervalTreeNode
        An interval tree over retention time containing all of the chromatograms in
        :attr:`chromatograms`
    """
    def __init__(self, chromatograms, sort=True):
        if sort:
            self.chromatograms = [c for c in sorted([c for c in chromatograms if len(c)], key=lambda x: (
                x.neutral_mass, x.start_time))]
        else:
            self.chromatograms = list(chromatograms)
        self._key_map = None
        self._intervals = None

    def _invalidate(self):
        self._key_map = None
        self._intervals = None

    def _build_key_map(self):
        self._key_map = defaultdict(list)
        for chrom in self:
            self._key_map[chrom.key].append(chrom)
        for key in self._key_map.keys():
            self._key_map[key] = DisjointChromatogramSet(self._key_map[key])

    def _build_rt_interval_tree(self):
        self._intervals = build_rt_interval_tree(self)

    @property
    def key_map(self):
        if self._key_map is None:
            self._build_key_map()
        return self._key_map

    @property
    def rt_interval_tree(self):
        if self._intervals is None:
            self._build_rt_interval_tree()
        return self._intervals

    def find_all_instances(self, key):
        if self._key_map is None:
            self._build_key_map()
        return self._key_map[key]

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        return self.chromatograms[i]

    def __len__(self):
        return len(self.chromatograms)

    def find_key(self, key):
        for obj in self:
            if obj.key == key:
                return obj

    def find_mass(self, mass, ppm_error_tolerance=1e-5):
        index, flag = binary_search_with_flag(self.chromatograms, mass, ppm_error_tolerance)
        if flag:
            return self[index]
        else:
            return None

    def find_all_by_mass(self, mass, ppm_error_tolerance=1e-5):
        width = mass * ppm_error_tolerance
        return self.mass_between(mass - width, mass + width)

    def _binary_search(self, mass, error_tolerance=1e-5):
        return binary_search_with_flag(self, mass, error_tolerance)

    def min_points(self, n=3, keep_if_msms=True):
        self.chromatograms = [c for c in self if (len(c) >= n) or c.has_msms]
        return self

    def split_sparse(self, delta_rt=1.):
        self.chromatograms = [
            seg for c in self
            for seg in c.split_sparse(delta_rt)
        ]
        return self

    def __repr__(self):
        return repr(list(self))

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.chromatograms)

    def __str__(self):
        return str(list(self))

    def spanning(self, rt):
        return self.__class__((c for c in self if c.start_time <= rt <= c.end_time), sort=False)

    def contained_in_interval(self, start, end):
        return self.__class__(
            (c for c in self if ((c.start_time <= start and c.end_time >= start) or (
                c.start_time >= start and c.end_time <= end) or (
                c.start_time >= start and c.end_time >= end and c.start_time <= end) or (
                c.start_time <= start and c.end_time >= start) or (
                c.start_time <= end and c.end_time >= end))), sort=False)

    def after(self, t):
        out = []
        for c in self:
            c = c.clone()
            c.truncate_before(t)
            if len(c) > 0:
                out.append(c)
        return self.__class__(out, sort=False)

    def before(self, t):
        out = []
        for c in self:
            c = c.clone()
            c.truncate_after(t)
            if len(c) > 0:
                out.append(c)
        return self.__class__(out, sort=False)

    def mass_between(self, low, high):
        n = len(self)
        if n == 0:
            return ChromatogramFilter([])
        low_index, flag = binary_search_with_flag(self.chromatograms, low, 1e-5)
        low_index = max(0, min(low_index, n - 1))
        if self[low_index].neutral_mass < low:
            low_index += 1
        high_index, flag = binary_search_with_flag(self.chromatograms, high, 1e-5)
        high_index += 2
        high_index = min(n - 1, high_index)
        if self[high_index].neutral_mass > high:
            high_index -= 1
        items = self[low_index:high_index]
        items = [c for c in items if low <= c.neutral_mass <= high]
        return ChromatogramFilter(items, sort=False)

    def filter(self, filter_fn):
        return self.__class__([x for x in self if filter_fn(x)], sort=False)

    @classmethod
    def process(cls, chromatograms, min_points=5, percentile=10, delta_rt=1.):
        return cls(chromatograms).split_sparse(delta_rt).min_points(min_points)

    def smooth_overlaps(self, mass_error_tolerance=1e-5):
        return self.__class__(smooth_overlaps(self, mass_error_tolerance))

    def extend(self, other):
        chroma = []
        chroma.extend(self)
        chroma.extend(other)
        self.chromatograms = [c for c in sorted([c for c in chroma if len(c)], key=lambda x: (
                              x.neutral_mass, x.start_time))]
        self._invalidate()

    def __add__(self, other):
        inst = self.__class__([])
        inst.extend(self)
        inst.extend(other)
        return inst


class DisjointChromatogramSet(object):
    def __init__(self, chromatograms):
        self.group = sorted(chromatograms, key=lambda c: c.start_time)

    def linear_search(self, start_time, end_time):
        center_time = (start_time + end_time) / 2.
        for chrom in self.group:
            if chrom.start_time <= center_time <= chrom.end_time:
                return chrom

    def find_overlap(self, chromatogram):
        return self.linear_search(
            chromatogram.start_time,
            chromatogram.end_time)

    def replace(self, original, replacement):
        i = self.group.index(original)
        self.group[i] = replacement

    def __getitem__(self, i):
        return self.group[i]

    def __iter__(self):
        return iter(self.group)

    def __repr__(self):
        return repr(list(self))

    def _repr_pretty_(self, p, cycle):
        return p.pretty(self.group)

    def __str__(self):
        return str(list(self))

    def __len__(self):
        return len(self.group)
