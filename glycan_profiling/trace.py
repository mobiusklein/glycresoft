import numpy as np
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict, OrderedDict, namedtuple
import itertools


dummyscan = namedtuple('dummyscan', ["id", "index", "scan_time"])


fake_scan = dummyscan("--not-a-real-scan--", -1, -1)


def group_by(ungrouped_list, key_fn=lambda x: x, transform_fn=lambda x: x):
    groups = defaultdict(list)
    for item in ungrouped_list:
        key_value = key_fn(item)
        groups[key_value].append(transform_fn(item))
    return groups


class MassShift(object):
    def __init__(self, name, mass):
        self.name = name
        self.mass = mass

    def __repr__(self):
        return "MassShift(%s, %f)" % (self.name, self.mass)

    def __mul__(self, n):
        if isinstance(n, int):
            return self.__class__("%d * %s" % (n, self.name), self.mass * n)
        else:
            raise TypeError("Cannot multiply MassShift by non-integer")


class Tracer(object):

    def __init__(self, scan_generator, database, mass_error_tolerance=1e-5):
        self.scan_generator = scan_generator
        self.database = database
        self.tracker = defaultdict(OrderedDict)
        self.mass_error_tolerance = mass_error_tolerance
        self.total_ion_chromatogram = SimpleChromatogram(self)
        self.base_peak_chromatogram = SimpleChromatogram(self)

    def configure_iteration(self, *args, **kwargs):
        self.scan_generator.configure_iteration(*args, **kwargs)

    def scan_id_to_rt(self, scan_id):
        return self.scan_generator.convert_scan_id_to_retention_time(scan_id)

    def _handle_generic_chromatograms(self, scan):
        self.total_ion_chromatogram[scan.id] = sum(p.intensity for p in scan)
        self.base_peak_chromatogram[scan.id] = max(p.intensity for p in scan)

    def next(self):
        idents = defaultdict(list)
        try:
            scan = next(self.scan_generator)
        except (ValueError, IndexError), e:
            print(e)
            return idents, fake_scan
        self._handle_generic_chromatograms(scan)
        for peak in scan.deconvoluted_peak_set:
            for match in self.database.search_mass_ppm(
                    peak.neutral_mass, self.mass_error_tolerance):
                self.tracker[match.serialize()].setdefault(scan.id, [])
                self.tracker[match.serialize()][scan.id].append(peak)
                idents[peak].append(match)
        return idents, scan

    def truncate_chromatograms(self, chromatograms):
        start, stop = find_truncation_points(*self.total_ion_chromatogram.as_arrays())
        out = []
        for c in chromatograms:
            if len(c) == 0:
                continue
            c.truncate_before(start)
            if len(c) == 0:
                continue
            c.truncate_after(stop)
            if len(c) == 0:
                continue
            out.append(c)
        return out

    def chromatograms(self):
        chroma = [
            Chromatogram(composition, observations.keys(), observations.values(), map(
                self.scan_generator.convert_scan_id_to_retention_time, observations))
            for composition, observations in self.tracker.items()
        ]
        chroma = self.truncate_chromatograms(chroma)
        return chroma


class IncludeUnmatchedTracer(Tracer):

    def __init__(self, scan_generator, database, mass_error_tolerance=1e-5):
        super(IncludeUnmatchedTracer, self).__init__(
            scan_generator, database, mass_error_tolerance)
        self.unmatched = []

    def __iter__(self):
        return self

    def next(self):
        idents = defaultdict(list)
        try:
            scan = next(self.scan_generator)
        except (ValueError, IndexError), e:
            print(e)
            return idents, fake_scan
        self._handle_generic_chromatograms(scan)
        for peak in scan.deconvoluted_peak_set:
            matches = self.database.search_mass_ppm(
                peak.neutral_mass, self.mass_error_tolerance)
            if matches:
                for match in matches:
                    self.tracker[match.serialize()].setdefault(scan.id, [])
                    self.tracker[match.serialize()][scan.id].append(peak)
                    idents[peak].append(match)
            else:
                self.unmatched.append((scan.id, peak))
        return idents, scan

    def __next__(self):
        return self.next()

    def chromatograms(self, minimum_mass=300, minimum_intensity=1000., grouping_tolerance=None):
        if grouping_tolerance is None:
            grouping_tolerance = self.mass_error_tolerance
        chroma = sorted(map(ChromatogramBuilder.from_chromatogram, (super(
            IncludeUnmatchedTracer, self).chromatograms())), key=lambda x: x.neutral_mass)
        assert is_sorted(chroma)
        unmatched = (sorted(self.unmatched, key=lambda x: x[
            1].intensity, reverse=True))
        for scan_id, peak in unmatched:
            if peak.neutral_mass < minimum_mass or peak.intensity < minimum_intensity:
                continue
            index, matched = binary_search_with_flag(
                chroma, peak.neutral_mass, grouping_tolerance)

            if matched:
                chroma[index].add(
                    scan_id, peak,
                    self.scan_generator.convert_scan_id_to_retention_time(scan_id))
            else:
                new = ChromatogramBuilder(peak.neutral_mass)
                new.add(
                    scan_id, peak,
                    self.scan_generator.convert_scan_id_to_retention_time(scan_id))
                if index != 0:
                    chroma.insert(index + 1, new)
                else:
                    x = chroma[index]
                    if x.neutral_mass < new.neutral_mass:
                        new_index = index + 1
                    else:
                        new_index = index
                    chroma.insert(new_index, new)

        chroma = [c.to_chromatogram() for c in chroma]
        chroma = self.truncate_chromatograms(chroma)
        return chroma


def is_sorted(mass_list):
    for a, b in zip(mass_list[:-1], mass_list[1:]):
        if not a.neutral_mass <= b.neutral_mass:
            print a.neutral_mass, b.neutral_mass
            return False
    return True


def binary_search_with_flag(array, mass, error_tolerance=1e-5):
    lo = 0
    hi = len(array)
    while hi != lo:
        mid = (hi + lo) / 2
        x = array[mid]
        err = (x.neutral_mass - mass) / mass
        if abs(err) <= error_tolerance:
            return mid, True
        elif (hi - lo) == 1:
            return mid, False
        elif err > 0:
            hi = mid
        elif err < 0:
            lo = mid
    return 0, False


def total_intensity(peaks):
    return sum(p.intensity for p in peaks)


class ChromatogramBuilder(object):

    @classmethod
    def from_chromatogram(cls, chroma):
        dd = defaultdict(list)
        peaks = chroma.peaks
        scan_ids = chroma.scan_ids
        retention_times = chroma.retention_times
        for peak, scan_id, rt in zip(peaks, scan_ids, retention_times):
            dd[rt].extend((peak_i, scan_id) for peak_i in peak)
        return cls(chroma.neutral_mass, dd, chroma.composition)

    def __init__(self, neutral_mass, mapping=None, composition=None):
        if mapping is None:
            mapping = defaultdict(list)
        self.neutral_mass = neutral_mass
        self.mapping = mapping
        self.composition = composition

    def add(self, scan_id, peak, retention_time):
        region = self.mapping[retention_time]
        region.append((peak, scan_id))

    def to_chromatogram(self):
        pairs = sorted(self.mapping.items())

        scan_ids = []
        peaks = []
        retention_times = []

        for rt, peak_scan_ids in pairs:
            by_scan_id = group_by(peak_scan_ids, lambda x: x[1])
            for scan_id, peaks_i in by_scan_id.items():
                peak_group = []
                map(peak_group.append, [x[0] for x in peaks_i])
                peaks.append(peak_group)
                scan_ids.append(scan_id)
                retention_times.append(rt)
        return Chromatogram(self.composition, scan_ids, peaks, retention_times)

    def __repr__(self):
        return "ChromatogramBuilder(%f, %d, %s)" % (self.neutral_mass, len(self.mapping), self.composition)


def base_removal(chromatogram, p=10):
    x, y = gaussian_filter1d(chromatogram.as_arrays(), 1)
    v = np.abs(np.diff(y))
    flag = v < np.percentile(y, p)
    return x[1:][~flag], y[1:][~flag]


def threshold_unpeaked(chromatogram):
    peak_data = base_removal(chromatogram)
    return len(peak_data[0]) > 3


def split_by_charge(peaks):
    return group_by(peaks, lambda x: x.charge)


def count_charge_states(peaks):
    peaks = [j for i in peaks for j in i]
    return len(split_by_charge(peaks))


class ChromatogramDeltaNode(object):
    def __init__(self, retention_times, delta_intensity, start_time, stop_time, is_below_threshold=True):
        self.retention_times = retention_times
        self.delta_intensity = delta_intensity
        self.start_time = start_time
        self.stop_time = stop_time
        self.mean_change = np.mean(delta_intensity)
        self.is_below_threshold = is_below_threshold

    def __repr__(self):
        return "ChromatogramDeltaNode(%f, %f, %f)" % (
            self.mean_change, self.start_time, self.stop_time)

    @classmethod
    def partition(cls, rt, delta_smoothed, window_size=.5):
        last_rt = rt[1]
        last_index = 1
        nodes = []
        for i, rt_i in enumerate(rt[2:]):
            if (rt_i - last_rt) >= window_size:
                nodes.append(
                    cls(
                        rt[last_index:i],
                        delta_smoothed[last_index:i + 1],
                        last_rt, rt[i]))
                last_index = i
                last_rt = rt_i
        nodes.append(
            cls(
                rt[last_index:i],
                delta_smoothed[last_index:i + 1],
                last_rt, rt[i]))
        return nodes


def find_truncation_points(rt, signal):
    rt = np.array(rt)
    smoothed = gaussian_filter1d(signal, 3)
    delta_smoothed = np.gradient(smoothed, rt)
    change = delta_smoothed[:-1] - delta_smoothed[1:]
    avg_change = change.mean()
    std_change = change.std()

    lo = avg_change - std_change
    hi = avg_change + std_change

    nodes = ChromatogramDeltaNode.partition(rt, delta_smoothed)
    for node in nodes:
        if lo > node.mean_change or node.mean_change > hi:
            node.is_below_threshold = False
    leading = 0
    ending = len(nodes)
    for node in nodes:
        if not node.is_below_threshold:
            break
        leading += 1
    for node in reversed(nodes):
        if not node.is_below_threshold:
            break
        ending -= 1
    ending = min(ending + 1, len(nodes) - 1)
    return nodes[leading].start_time, nodes[ending].start_time


class SimpleChromatogram(OrderedDict):
    def __init__(self, time_converter):
        self.time_converter = time_converter
        super(SimpleChromatogram, self).__init__()

    def as_arrays(self):
        return (
            np.array(map(self.time_converter.scan_id_to_rt, self)),
            np.array(self.values()))


class Chromatogram(object):

    def __init__(self, composition, scan_ids, peaks, retention_times=None, adducts=None, used_as_adduct=False):
        if adducts is None:
            adducts = []
        self.composition = composition
        self.peaks = peaks
        self.scan_ids = scan_ids
        self.retention_times = retention_times
        self.total_signal = sum(map(total_intensity, self.peaks))
        self.neutral_mass = max(itertools.chain.from_iterable(self.peaks),
                                key=lambda x: x.intensity).neutral_mass
        self.charge_states = split_by_charge([p_i for p in peaks for p_i in p]).keys()
        self.n_charge_states = len(self.charge_states)
        self.adducts = adducts
        self.used_as_adduct = used_as_adduct

    @property
    def key(self):
        if self.composition is not None:
            return self.composition
        else:
            return self.neutral_mass

    @property
    def start_time(self):
        return self.retention_times[0]

    @property
    def end_time(self):
        return self.retention_times[-1]

    def as_arrays(self):
        return map(np.array, [self.retention_times, map(total_intensity, self.peaks)])

    def __len__(self):
        return len(self.scan_ids)

    def __repr__(self):
        return "Chromatogram(%s, %0.4f)" % (self.composition, self.neutral_mass)

    def __iter__(self):
        for i in range(len(self)):
            yield self.scan_ids[i], self.peaks[i], self.retention_times[i]

    def split_sparse(self, delta_rt=1.):
        chunks = []
        current_scan_ids = []
        current_peaks = []
        current_rts = []
        groups = zip(self.scan_ids, self.peaks, self.retention_times)
        scan_id, peaks, rt = groups[0]
        current_scan_ids.append(scan_id)
        current_peaks.append(peaks)
        last_rt = rt
        current_rts.append(rt)

        for scan_id, peaks, rt in groups[1:]:
            if rt - last_rt > delta_rt:
                chunk = Chromatogram(
                    self.composition, current_scan_ids, current_peaks, current_rts)
                chunks.append(chunk)
                current_scan_ids = []
                current_peaks = []
                current_rts = []
            current_scan_ids.append(scan_id)
            current_peaks.append(peaks)
            last_rt = rt
            current_rts.append(rt)
        chunk = Chromatogram(
            self.composition, current_scan_ids, current_peaks, current_rts)
        chunk.adducts = list(self.adducts)
        chunk.used_as_adduct = self.used_as_adduct
        chunks.append(chunk)
        return chunks

    def truncate_before(self, time):
        for i, rt in enumerate(self.retention_times):
            if rt >= time:
                break
        self.scan_ids = self.scan_ids[i + 1:]
        self.peaks = self.peaks[i + 1:]
        self.retention_times = self.retention_times[i + 1:]

    def truncate_after(self, time):
        for i, rt in enumerate(self.retention_times):
            if rt >= time:
                break
        self.scan_ids = self.scan_ids[:i]
        self.peaks = self.peaks[:i]
        self.retention_times = self.retention_times[:i]

    def merge(self, other):
        builder = ChromatogramBuilder.from_chromatogram(self)
        for scan_id, peaks, rt in other:
            for peak in peaks:
                builder.add(scan_id, peak, rt)
        new = builder.to_chromatogram()
        new.adducts = list(self.adducts)
        new.used_as_adduct = self.used_as_adduct
        return new

    def slice(self, start, end):
        builder = ChromatogramBuilder()
        for scan_id, peaks, rt in self:
            if start <= rt <= end:
                for peak in peaks:
                    builder.add(scan_id, peak, rt)
        new = builder.to_chromatogram()
        new = builder.to_chromatogram()
        new.adducts = list(self.adducts)
        new.used_as_adduct = self.used_as_adduct
        return new


class ChromatogramFilter(object):
    def __init__(self, chromatograms):
        self.chromatograms = [c for c in sorted(chromatograms, key=lambda x: x.neutral_mass) if len(c)]

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
        index, flag = binary_search_with_flag(self, mass, ppm_error_tolerance)
        if flag:
            return self[index]
        else:
            return None

    def min_points(self, n=3):
        self.chromatograms = [c for c in self if len(c) >= n]
        return self

    def threshold_unpeaked(self, n=3, p=10):
        self.chromatograms = [
            c for c in self
            if len(base_removal(c, p)[0]) >= n
        ]
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
        return self.__class__(c for c in self if c.start_time < rt < c.end_time)

    @classmethod
    def process(cls, chromatograms, n_peaks=5, percentile=10, delta_rt=1.):
        return cls(chromatograms).split_sparse(delta_rt).min_points(n_peaks)  # .threshold_unpeaked(n_peaks, percentile)


def span_overlap(self, interval):
    cond = ((self.start_time <= interval.start_time and self.end_time >= interval.end_time) or (
        self.start_time >= interval.start_time and self.end_time <= interval.end_time) or (
        self.start_time >= interval.start_time and self.end_time >= interval.end_time))
    return cond


def join_mass_shifted(chromatograms, adducts, mass_error_tolerance=1e-5):
    out = []
    for chroma in chromatograms:
        add = chroma
        for adduct in adducts:
            match = chromatograms.find_mass(chroma.neutral_mass + adduct.mass, mass_error_tolerance)
            if match and span_overlap(add, match):
                match.used_as_adduct = True
                add = add.merge(match)
                add.adducts.append(adduct)
        out.append(add)
    return ChromatogramFilter(out)
