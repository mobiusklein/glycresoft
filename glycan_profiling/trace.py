import os
import tempfile
import numpy as np
from scipy.ndimage import gaussian_filter1d

from collections import defaultdict, OrderedDict, namedtuple

from ms_deisotope.output.db import DatabaseScanSerializer, ScanBunch, DatabaseScanDeserializer

from .chromatogram_tree import (
    Chromatogram, ChromatogramForest, Unmodified,
    mask_subsequence)

from .scan_cache import (
    DatabaseScanCacheHandler, NullScanCacheHandler, ThreadedDatabaseScanCacheHandler)

dummyscan = namedtuple('dummyscan', ["id", "index", "scan_time"])


fake_scan = dummyscan("--not-a-real-scan--", -1, -1)


class Tracer(object):
    def __init__(self, scan_generator, database, mass_error_tolerance=1e-5,
                 cache_handler_type=ThreadedDatabaseScanCacheHandler):
        self.scan_generator = scan_generator

        self.database = database

        self.tracker = defaultdict(OrderedDict)
        self.mass_error_tolerance = mass_error_tolerance

        self.total_ion_chromatogram = SimpleChromatogram(self)
        self.base_peak_chromatogram = SimpleChromatogram(self)

        self.scan_store = None
        self._scan_store_type = cache_handler_type

        self.configure_cache()

    @property
    def scan_source(self):
        return self.scan_generator.scan_source

    def configure_cache(self):
        self.scan_store = self._scan_store_type.configure_storage(self.scan_source)

    def configure_iteration(self, *args, **kwargs):
        self.scan_generator.configure_iteration(*args, **kwargs)

    def scan_id_to_rt(self, scan_id):
        return self.scan_generator.convert_scan_id_to_retention_time(scan_id)

    def _handle_generic_chromatograms(self, scan):
        self.total_ion_chromatogram[scan.id] = sum(p.intensity for p in scan)
        self.base_peak_chromatogram[scan.id] = max(p.intensity for p in scan)

    def store_scan(self, scan):
        self.scan_store.accumulate(scan)

    def commit(self):
        self.scan_store.commit()

    def next_scan(self):
        scan = next(self.scan_generator)
        self.store_scan(scan)
        while scan.ms_level != 1:
            scan = next(self.scan_generator)
            self.store_scan(scan)
        return scan

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        idents = defaultdict(list)
        try:
            scan = self.next_scan()
            self._handle_generic_chromatograms(scan)
        except (ValueError, IndexError), e:
            print(e)
            return idents, fake_scan
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

    def find_truncation_points(self):
        start, stop = find_truncation_points(*self.total_ion_chromatogram.as_arrays())
        return start, stop

    def chromatograms(self, truncate=True):
        chroma = [
            Chromatogram.from_parts(composition, map(
                self.scan_id_to_rt, observations), observations.keys(),
                observations.values())
            for composition, observations in self.tracker.items()
        ]
        if truncate:
            chroma = self.truncate_chromatograms(chroma)
        return chroma


class IncludeUnmatchedTracer(Tracer):

    def __init__(self, scan_generator, database, mass_error_tolerance=1e-5):
        super(IncludeUnmatchedTracer, self).__init__(
            scan_generator, database, mass_error_tolerance)
        self.unmatched = []

    def next(self):
        idents = defaultdict(list)
        try:
            scan = self.next_scan()
            self._handle_generic_chromatograms(scan)
        except (ValueError, IndexError), e:
            print(e)
            return idents, fake_scan
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

    def chromatograms(self, minimum_mass=300, minimum_intensity=1000., grouping_tolerance=None, truncate=True):
        if grouping_tolerance is None:
            grouping_tolerance = self.mass_error_tolerance
        chroma = sorted(super(
            IncludeUnmatchedTracer, self).chromatograms(truncate=True), key=lambda x: x.neutral_mass)
        forest = ChromatogramForest(chroma, grouping_tolerance, self.scan_id_to_rt)
        forest.aggregate_unmatched_peaks(self.unmatched, minimum_mass, minimum_intensity)
        chroma = list(forest)
        if truncate:
            chroma = self.truncate_chromatograms(chroma)
        return chroma


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


class ChromatogramDeltaNode(object):
    def __init__(self, retention_times, delta_intensity, start_time, end_time, is_below_threshold=True):
        self.retention_times = retention_times
        self.delta_intensity = delta_intensity
        self.start_time = start_time
        self.end_time = end_time
        self.mean_change = np.mean(delta_intensity)
        self.is_below_threshold = is_below_threshold

    def __repr__(self):
        return "ChromatogramDeltaNode(%f, %f, %f)" % (
            self.mean_change, self.start_time, self.end_time)

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


def build_chromatogram_nodes(rt, signal):
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

    return nodes


def find_truncation_points(rt, signal):
    nodes = build_chromatogram_nodes(rt, signal)

    leading = 0
    ending = len(nodes)

    for node in nodes:
        if not node.is_below_threshold:
            break
        leading += 1
    leading -= 3
    leading = max(leading, 0)

    for node in reversed(nodes):
        if not node.is_below_threshold:
            break
        ending -= 1

    ending = min(ending + 2, len(nodes) - 1)
    if len(nodes) == 1:
        return nodes[0].start_time, nodes[0].end_time
    elif len(nodes) == 2:
        return nodes[0].start_time, nodes[-1].end_time
    return nodes[leading].start_time, nodes[ending].end_time


class SimpleChromatogram(OrderedDict):
    def __init__(self, time_converter):
        self.time_converter = time_converter
        super(SimpleChromatogram, self).__init__()

    def as_arrays(self):
        return (
            np.array(map(self.time_converter.scan_id_to_rt, self)),
            np.array(self.values()))


class ChromatogramFilter(object):
    def __init__(self, chromatograms, sort=True):
        if sort:
            self.chromatograms = [c for c in sorted([c for c in chromatograms if len(c)], key=lambda x: x.neutral_mass)]
        else:
            self.chromatograms = list(chromatograms)

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
        return self.__class__((c for c in self if c.start_time < rt < c.end_time), sort=False)

    def mass_between(self, low, high):
        low_index, flag = binary_search_with_flag(self, low, 1e-5)
        if self[low_index] < low:
            low_index += 1
        high_index, flag = binary_search_with_flag(self, high, 1e-5)
        if self[high_index] > high:
            high_index -= 1
        return ChromatogramFilter(self[low_index:high_index], sort=False)

    def filter(self, filter_fn):
        return self.__class__([x for x in self if filter_fn(x)], sort=False)

    @classmethod
    def process(cls, chromatograms, n_peaks=5, percentile=10, delta_rt=1.):
        return cls(chromatograms).split_sparse(delta_rt).min_points(n_peaks)


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
                match.used_as_adduct.append((add.key, adduct))
                add = add.merge(match, node_type=adduct)
                add.created_at = "join_mass_shifted"
                add.adducts.append(adduct)
        out.append(add)
    return ChromatogramFilter(out)


def reverse_adduction_search(chromatograms, adducts, mass_error_tolerance, database):
    exclude_compositions = dict()
    candidate_chromatograms = []

    new_members = {}
    unmatched = []

    for chroma in chromatograms:
        if chroma.composition is not None:
            exclude_compositions[chroma.composition] = chroma
        else:
            candidate_chromatograms.append(chroma)

    for chroma in candidate_chromatograms:
        candidate_mass = chroma.neutral_mass
        matched = False
        exclude = False
        for adduct in adducts:
            matches = database.search_mass_ppm(
                candidate_mass - adduct.mass, mass_error_tolerance)
            for match in matches:
                name = str(match)
                if name in exclude_compositions:
                    exclude = True
                    continue
                if name in new_members:
                    chroma_to_update = new_members[name]
                else:
                    chroma_to_update = Chromatogram(name)
                    chroma_to_update.created_at = "reverse_adduction_search"
                chroma, _ = chroma.bisect_adduct(Unmodified)
                chroma_to_update = chroma_to_update.merge(chroma, adduct)
                chroma_to_update.created_at = "reverse_adduction_search"
                new_members[name] = chroma_to_update
                matched = True
        if not matched and not exclude:
            unmatched.append(chroma)
    out = []
    out.extend(exclude_compositions.values())
    out.extend(new_members.values())
    out.extend(unmatched)
    return ChromatogramFilter(out)


def prune_bad_adduct_branches(solutions):
    key_map = {c.key: c for c in solutions}
    updated = set()
    for case in solutions:
        if case.used_as_adduct:
            keepers = []
            for owning_key, adduct in case.used_as_adduct:
                owner = key_map.get(owning_key)
                if owner is None:
                    continue
                if case.score > owner.score:
                    new_masked = mask_subsequence(owner, case)
                    new_masked.created_at = "prune_bad_adduct_branches"
                    key_map[owning_key] = new_masked
                    new_masked.score = owner.score
                    updated.add(owning_key)
                else:
                    keepers.append((owning_key, adduct))
            case.chromatogram.used_as_adduct = keepers
    out = [key_map[k].chromatogram for k in set(key_map) - updated]
    out.extend(key_map[k] for k in updated)
    return ChromatogramFilter(out)
