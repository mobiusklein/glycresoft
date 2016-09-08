from collections import defaultdict, OrderedDict, namedtuple

from .chromatogram_tree import (
    Chromatogram, ChromatogramForest, Unmodified,
    mask_subsequence, DuplicateNodeError, smooth_overlaps,
    SimpleChromatogram, find_truncation_points, build_rt_interval_tree)

from .scan_cache import (
    NullScanCacheHandler, ThreadedDatabaseScanCacheHandler)

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

    @property
    def scan_source(self):
        try:
            return self.scan_generator.scan_source
        except AttributeError:
            return None

    def configure_cache(self, storage_path=None, name=None):
        if storage_path is None:
            storage_path = self.scan_source
        self.scan_store = self._scan_store_type.configure_storage(storage_path, name)

    def configure_iteration(self, *args, **kwargs):
        self.scan_generator.configure_iteration(*args, **kwargs)

    def scan_id_to_rt(self, scan_id):
        return self.scan_generator.convert_scan_id_to_retention_time(scan_id)

    def _handle_generic_chromatograms(self, scan):
        tic = sum(p.intensity for p in scan)
        self.total_ion_chromatogram[scan.id] = tic
        self.base_peak_chromatogram[scan.id] = max(p.intensity for p in scan) if tic > 0 else 0

    def store_scan(self, scan):
        if self.scan_store is not None:
            self.scan_store.accumulate(scan)

    def commit(self):
        if self.scan_store is not None:
            self.scan_store.commit()

    def complete(self):
        if self.scan_store is not None:
            self.scan_store.complete()
        self.scan_generator.close()

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

    def __init__(self, scan_generator, database, mass_error_tolerance=1e-5,
                 cache_handler_type=ThreadedDatabaseScanCacheHandler):
        super(IncludeUnmatchedTracer, self).__init__(
            scan_generator, database, mass_error_tolerance, cache_handler_type=cache_handler_type)
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
            IncludeUnmatchedTracer, self).chromatograms(truncate=truncate), key=lambda x: x.neutral_mass)
        forest = ChromatogramForest(chroma, grouping_tolerance, self.scan_id_to_rt)
        forest.aggregate_unmatched_peaks(self.unmatched, minimum_mass, minimum_intensity)
        chroma = list(forest)
        if truncate:
            chroma = self.truncate_chromatograms(chroma)
        return chroma


def binary_search_with_flag(array, mass, error_tolerance=1e-5):
    lo = 0
    n = hi = len(array)
    while hi != lo:
        mid = (hi + lo) / 2
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
    def __init__(self, chromatograms, sort=True):
        if sort:
            self.chromatograms = [c for c in sorted([c for c in chromatograms if len(c)], key=lambda x: (
                x.neutral_mass, x.start_time))]
        else:
            self.chromatograms = list(chromatograms)
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

    def _binary_search(self, mass, error_tolerance=1e-5):
        return binary_search_with_flag(self, mass, error_tolerance)

    def _sweep_find_mass(self, mass, error_tolerance=1e-5):
        low = mass - (mass * error_tolerance)
        high = mass + (mass * error_tolerance)
        cases = self.mass_between(low, high)
        if len(cases) == 0:
            return None
        best_index = 0
        best_error = float('inf')

        for i, case in enumerate(cases):
            err = abs((case.neutral_mass - mass) / mass)
            if err < best_error and err < error_tolerance:
                best_error = err
                best_index = i
        return cases[best_index]

    def min_points(self, n=3, keep_if_msms=True):
        self.chromatograms = [c for c in self if len(c) >= n or c.has_msms]
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
        low_index, flag = binary_search_with_flag(self.chromatograms, low, 1e-5)
        if self[low_index].neutral_mass < low:
            low_index += 1
        high_index, flag = binary_search_with_flag(self.chromatograms, high, 1e-5)
        high_index += 2
        if self[high_index].neutral_mass > high:
            high_index -= 1
        return ChromatogramFilter(self[low_index:high_index], sort=False)

    def filter(self, filter_fn):
        return self.__class__([x for x in self if filter_fn(x)], sort=False)

    @classmethod
    def process(cls, chromatograms, n_peaks=5, percentile=10, delta_rt=1.):
        return cls(chromatograms).split_sparse(delta_rt).min_points(n_peaks)

    def smooth_overlaps(self, mass_error_tolerance=1e-5):
        return self.__class__(smooth_overlaps(self, mass_error_tolerance))


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


def span_overlap(self, interval):
    cond = ((self.start_time <= interval.start_time and self.end_time >= interval.end_time) or (
        self.start_time >= interval.start_time and self.end_time <= interval.end_time) or (
        self.start_time >= interval.start_time and self.end_time >= interval.end_time and
        self.start_time <= interval.end_time) or (
        self.start_time <= interval.start_time and self.end_time >= interval.start_time) or (
        self.start_time <= interval.end_time and self.end_time >= interval.end_time))
    return cond


def join_mass_shifted(chromatograms, adducts, mass_error_tolerance=1e-5):
    out = []
    for chroma in chromatograms:
        add = chroma
        for adduct in adducts:
            match = chromatograms.find_mass(chroma.neutral_mass + adduct.mass, mass_error_tolerance)
            if match and span_overlap(add, match):
                try:
                    match.used_as_adduct.append((add.key, adduct))
                    add = add.merge(match, node_type=adduct)
                    add.created_at = "join_mass_shifted"
                    add.adducts.append(adduct)
                except DuplicateNodeError, e:
                    e.original = chroma
                    e.to_add = match
                    e.accumulated = add
                    e.adduct = adduct
                    raise e
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
    solutions._build_key_map()
    key_map = solutions._key_map
    updated = set()
    for case in solutions:
        if case.used_as_adduct:
            keepers = []
            for owning_key, adduct in case.used_as_adduct:
                owner = key_map.get(owning_key)
                if owner is None:
                    continue
                owner_item = owner.find_overlap(case)
                if owner_item is None:
                    continue
                if case.score > owner_item.score:
                    new_masked = mask_subsequence(owner_item, case)
                    new_masked.created_at = "prune_bad_adduct_branches"
                    new_masked.score = owner_item.score
                    if len(new_masked) != 0:
                        owner.replace(owner_item, new_masked)
                    updated.add(owning_key)
                else:
                    keepers.append((owning_key, adduct))
            case.chromatogram.used_as_adduct = keepers
    out = [s.chromatogram for k in (set(key_map) - updated) for s in key_map[k]]
    out.extend(s for k in updated for s in key_map[k])
    return ChromatogramFilter(out)
