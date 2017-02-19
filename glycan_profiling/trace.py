import time
from collections import defaultdict

import numpy as np

from glycan_profiling.task import TaskBase

from .chromatogram_tree import (
    Chromatogram, ChromatogramForest, Unmodified,
    mask_subsequence, DuplicateNodeError, get_chromatogram,
    SimpleChromatogram, find_truncation_points,
    ChromatogramFilter, GlycanCompositionChromatogram, GlycopeptideChromatogram,
    ChromatogramOverlapSmoother)

from .scan_cache import NullScanCacheHandler

from .scoring import (
    ChromatogramSolution, NetworkScoreDistributor, ChromatogramScorer)


class ScanSink(object):
    def __init__(self, scan_generator, cache_handler_type=NullScanCacheHandler):
        self.scan_generator = scan_generator
        self.scan_store = None
        self._scan_store_type = cache_handler_type

    @property
    def scan_source(self):
        try:
            return self.scan_generator.scan_source
        except AttributeError:
            return None

    @property
    def sample_run(self):
        try:
            return self.scan_store.sample_run
        except AttributeError:
            return None

    def configure_cache(self, storage_path=None, name=None, source=None):
        self.scan_store = self._scan_store_type.configure_storage(
            storage_path, name, source)

    def configure_iteration(self, *args, **kwargs):
        self.scan_generator.configure_iteration(*args, **kwargs)

    def scan_id_to_rt(self, scan_id):
        return self.scan_generator.convert_scan_id_to_retention_time(scan_id)

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
        return self.next_scan()

    def next(self):
        return self.next_scan()


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
                except DuplicateNodeError as e:
                    e.original = chroma
                    e.to_add = match
                    e.accumulated = add
                    e.adduct = adduct
                    raise e
        out.append(add)
    return ChromatogramFilter(out)


def reverse_adduction_search(chromatograms, adducts, mass_error_tolerance, database, chromatogram_type=None):
    if chromatogram_type is None:
        chromatogram_type = GlycanCompositionChromatogram
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
                    chroma_to_update = chromatogram_type(match)
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
                    new_masked = mask_subsequence(get_chromatogram(owner_item), get_chromatogram(case))
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


class CompositionGroup(object):
    def __init__(self, name, members):
        self.name = name
        self.members = tuple(members)

    def __iter__(self):
        return iter(self.members)

    def __repr__(self):
        return "CompositionGroup(%r, %d members)" % (self.name, len(self.members))

    def __eq__(self, other):
        return self.members == other.members


class ChromatogramMatcher(TaskBase):
    def __init__(self, database, chromatogram_type=None):
        if chromatogram_type is None:
            chromatogram_type = GlycanCompositionChromatogram
        self.database = database
        self._group_bundle = dict()
        self.chromatogram_type = chromatogram_type

    def _match(self, neutral_mass, mass_error_tolerance=1e-5):
        return self.database.search_mass_ppm(neutral_mass, mass_error_tolerance)

    def _prepare_group(self, key, matches):
        ids = frozenset(m.id for m in matches)
        if len(ids) == 0:
            return None
        try:
            bundle = self._group_bundle[ids]
            return bundle
        except KeyError:
            bundle = CompositionGroup(key, [
                self.database._convert(m)
                for m in sorted(matches, key=lambda x: x.calculated_mass)])
            self._group_bundle[ids] = bundle
            return bundle

    def match(self, mass, mass_error_tolerance=1e-5):
        hits = self._match(mass, mass_error_tolerance)
        bundle = self._prepare_group(mass, hits)
        return bundle

    def assign(self, chromatogram, group):
        out = []
        if group is None:
            return [chromatogram]
        for composition in group:
            case = chromatogram.clone(self.chromatogram_type)
            case.composition = composition
            out.append(case)
        if len(out) == 0:
            return [chromatogram]
        return out

    def search(self, chromatogram, mass_error_tolerance=1e-5):
        return self.assign(chromatogram, self.match(
            chromatogram.weighted_neutral_mass, mass_error_tolerance))

    def reverse_adduct_search(self, chromatograms, adducts, mass_error_tolerance=1e-5):
        exclude_compositions = defaultdict(list)
        candidate_chromatograms = []

        new_members = {}
        unmatched = []

        for chroma in chromatograms:
            if chroma.composition is not None:
                exclude_compositions[chroma.composition].append(chroma)
            else:
                candidate_chromatograms.append(chroma)

        for chroma in candidate_chromatograms:
            candidate_mass = chroma.weighted_neutral_mass
            matched = False
            exclude = False
            for adduct in adducts:
                matches = self.match(candidate_mass - adduct.mass, mass_error_tolerance)
                if matches is None:
                    continue
                for match in matches:
                    name = (match)
                    if name in exclude_compositions:
                        exclude = True
                        continue
                    if name in new_members:
                        chroma_to_update = new_members[name]
                    else:
                        chroma_to_update = self.chromatogram_type(match)
                        chroma_to_update.created_at = "reverse_adduction_search"
                    chroma, _ = chroma.bisect_adduct(Unmodified)
                    chroma_to_update = chroma_to_update.merge(chroma, adduct)
                    chroma_to_update.created_at = "reverse_adduction_search"
                    new_members[name] = chroma_to_update
                    matched = True
            if not matched and not exclude:
                unmatched.append(chroma)
        out = []
        out.extend(s for g in exclude_compositions.values() for s in g)
        out.extend(new_members.values())
        out.extend(unmatched)
        return ChromatogramFilter(out)

    def join_mass_shifted(self, chromatograms, adducts, mass_error_tolerance=1e-5):
        out = []
        i = 0
        for chroma in chromatograms:
            add = chroma
            for adduct in adducts:
                match = chromatograms.find_mass(chroma.weighted_neutral_mass + adduct.mass, mass_error_tolerance)
                if match and span_overlap(add, match):
                    try:
                        match.used_as_adduct.append((add.key, adduct))
                        add = add.merge(match, node_type=adduct)
                        add.created_at = "join_mass_shifted"
                        add.adducts.append(adduct)
                    except DuplicateNodeError as e:
                        e.original = chroma
                        e.to_add = match
                        e.accumulated = add
                        e.adduct = adduct
                        raise e
            out.append(add)
            i += 1
        return ChromatogramFilter(out)

    def join_common_identities(self, chromatograms):
        chromatograms._build_key_map()
        key_map = chromatograms._key_map
        out = []
        for key, disjoint_set in key_map.items():
            if len(tuple(disjoint_set)) == 1:
                out.extend(disjoint_set)
                continue

            accumulated = []
            last = disjoint_set[0]
            for case in disjoint_set[1:]:
                if last.overlaps_in_time(case):
                    last = last._merge_missing_only(case)
                    last.created_at = "join_common_identities"
                else:
                    accumulated.append(last)
                    last = case
            accumulated.append(last)
            out.extend(accumulated)
        return ChromatogramFilter(out)

    def process(self, chromatograms, adducts=None, mass_error_tolerance=1e-5):
        if adducts is None:
            adducts = []
        matches = []
        chromatograms = list(chromatograms)
        self.log("Matching chromatograms")
        i = 0
        n = len(chromatograms)
        for chro in chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("%0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            matches.extend(self.search(chro, mass_error_tolerance))
        matches = ChromatogramFilter(matches)
        matches = self.join_common_identities(matches)
        matches = self.join_mass_shifted(matches, adducts, mass_error_tolerance)
        self.log("Handling Adducts")
        matches = self.reverse_adduct_search(matches, adducts, mass_error_tolerance)
        matches = self.join_common_identities(matches)
        return matches


class GlycanChromatogramMatcher(ChromatogramMatcher):
    pass


class GlycopeptideChromatogramMatcher(ChromatogramMatcher):
    def __init__(self, database, chromatogram_type=None):
        if chromatogram_type is None:
            chromatogram_type = GlycopeptideChromatogram
        super(GlycopeptideChromatogramMatcher).__init__(database, chromatogram_type)


class NonSplittingChromatogramMatcher(ChromatogramMatcher):
    def __init__(self, database, chromatogram_type=None):
        if chromatogram_type is None:
            chromatogram_type = Chromatogram
        super(NonSplittingChromatogramMatcher, self).__init__(
            database, chromatogram_type)

    def assign(self, chromatogram, group):
        out = []
        if group is None:
            return [chromatogram]
        else:
            case = chromatogram.clone(self.chromatogram_type)
            case.composition = group
            out.append(case)
            return out


class ChromatogramEvaluator(TaskBase):
    def __init__(self, scoring_model=None, network=None):
        if scoring_model is None:
            scoring_model = ChromatogramScorer()
        self.scoring_model = scoring_model
        self.network = network

    def evaluate(self, chromatograms, base_coef=0.8, support_coef=0.2, rt_delta=0.25,
                 min_points=3, smooth=True):
        filtered = ChromatogramFilter.process(
            chromatograms, delta_rt=rt_delta, min_points=min_points)
        if smooth:
            filtered = ChromatogramOverlapSmoother(filtered)

        solutions = []
        i = 0
        n = len(filtered)
        for case in filtered:
            start = time.time()
            i += 1
            if i % 1000 == 0:
                self.log("%0.2f%% chromatograms evaluated (%d/%d)" % (i * 100. / n, i, n))
            try:
                solutions.append(ChromatogramSolution(case, scorer=self.scoring_model))
                end = time.time()
                # Report on anything that took more than 30 seconds to evaluate
                if end - start > 30.0:
                    self.log("%r took a long time to evaluated (%0.2fs)" % (case, end - start))
            except (IndexError, ValueError):
                continue
        if base_coef != 1.0 and self.network is not None:
            NetworkScoreDistributor(solutions, self.network).distribute(base_coef, support_coef)
        return ChromatogramFilter(solutions)

    def score(self, chromatograms, base_coef=0.8, support_coef=0.2, rt_delta=0.25, min_points=3,
              smooth=True, adducts=None):

        solutions = self.evaluate(chromatograms, base_coef, support_coef, rt_delta, min_points, smooth)

        if adducts is not None and len(adducts):
            hold = prune_bad_adduct_branches(ChromatogramFilter(solutions))
            self.log("Re-evaluating after adduct pruning")
            solutions = self.evaluate(hold, base_coef, support_coef, rt_delta)

        solutions = ChromatogramFilter(sol for sol in solutions if sol.score > 1e-5)
        return solutions

    def acceptance_filter(self, solutions, threshold=0.4):
        return ChromatogramFilter([
            sol for sol in solutions
            if sol.score >= threshold and not sol.used_as_adduct
        ])


class ChromatogramExtractor(TaskBase):
    def __init__(self, peak_loader, truncate=False, minimum_mass=500, grouping_tolerance=1.5e-5,
                 minimum_intensity=250., min_points=3, delta_rt=0.25):
        self.peak_loader = peak_loader
        self.truncate = truncate
        self.minimum_mass = minimum_mass
        self.minimum_intensity = minimum_intensity
        self.grouping_tolerance = grouping_tolerance

        self.min_points = min_points
        self.delta_rt = delta_rt

        self.accumulated = None
        self.annotated_peaks = None
        self.peak_mapping = None

        self.chromatograms = None
        self.base_peak_chromatogram = None
        self.total_ion_chromatogram = None

    def get_scan_by_id(self, scan_id):
        return self.peak_loader.get_scan_by_id(scan_id)

    def get_scan_header_by_id(self, scan_id):
        return self.peak_loader.get_scan_header_by_id(scan_id)

    def get_index_information_by_scan_id(self, scan_id):
        return self.peak_loader.get_index_information_by_scan_id(scan_id)

    def scan_id_to_rt(self, scan_id):
        return self.peak_loader.convert_scan_id_to_retention_time(scan_id)

    def load_peaks(self):
        self.accumulated = self.peak_loader.ms1_peaks_above(self.minimum_mass)
        self.annotated_peaks = [x[:2] for x in self.accumulated]
        self.peak_mapping = {x[:2]: x[2] for x in self.accumulated}
        self.minimum_intensity = np.percentile([p[1].intensity for p in self.accumulated], 5)

    def aggregate_chromatograms(self):
        forest = ChromatogramForest([], self.grouping_tolerance, self.scan_id_to_rt)
        forest.aggregate_peaks(self.annotated_peaks, self.minimum_mass, self.minimum_intensity)
        chroma = list(forest)
        self.log("%d Chromatograms Extracted." % (len(chroma),))
        self.chromatograms = ChromatogramFilter.process(
            chroma, min_points=self.min_points, delta_rt=self.delta_rt)

    def summary_chromatograms(self):
        mapping = defaultdict(list)
        for scan_id, peak in self.annotated_peaks:
            mapping[scan_id].append(peak.intensity)
        bpc = SimpleChromatogram(self)
        tic = SimpleChromatogram(self)
        collection = sorted(mapping.items(), key=lambda b: self.scan_id_to_rt(b[0]))
        for scan_id, intensities in collection:
            bpc[scan_id] = max(intensities)
            tic[scan_id] = sum(intensities)
        self.base_peak_chromatogram = bpc
        self.total_ion_chromatogram = tic

    def run(self):
        self.log("... Begin Extracting Chromatograms")
        self.load_peaks()
        self.aggregate_chromatograms()
        self.summary_chromatograms()
        if self.truncate:
            self.chromatograms = ChromatogramFilter(
                self.truncate_chromatograms(self.chromatograms))
        return self.chromatograms

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

    def __iter__(self):
        if self.chromatograms is None:
            self.run()
        return iter(self.chromatograms)


class ChromatogramProcessor(TaskBase):
    matcher_type = GlycanChromatogramMatcher

    def __init__(self, chromatograms, database, adducts=None, mass_error_tolerance=1e-5,
                 scoring_model=None, network_sharing=0.,
                 smooth=True, acceptance_threshold=0.4):
        if adducts is None:
            adducts = []
        self._chromatograms = (chromatograms)
        self.database = database
        self.adducts = adducts
        self.mass_error_tolerance = mass_error_tolerance
        self.scoring_model = scoring_model
        self.network = database.glycan_composition_network
        self.base_coef = 1 - network_sharing
        self.support_coef = network_sharing
        self.smooth = smooth
        self.acceptance_threshold = acceptance_threshold

        self.solutions = None
        self.accepted_solutions = None

    def run(self):
        self.log("Begin Matching Chromatograms")
        matcher = self.matcher_type(self.database)
        matches = matcher.process(self._chromatograms, self.adducts, self.mass_error_tolerance)
        self.log("End Matching Chromatograms")
        self.log("Begin Evaluating Chromatograms")
        evaluator = ChromatogramEvaluator(self.scoring_model, self.network)
        self.solutions = evaluator.score(
            matches, self.base_coef, self.support_coef,
            smooth=self.smooth, adducts=self.adducts)
        self.accepted_solutions = evaluator.acceptance_filter(self.solutions)
        self.log("End Evaluating Chromatograms")

    def __iter__(self):
        if self.accepted_solutions is None:
            self.run()
        return iter(self.accepted_solutions)


class GlycopeptideChromatogramProcessor(ChromatogramProcessor):
    matcher_type = GlycopeptideChromatogramMatcher


class NonSplittingChromatogramProcessor(ChromatogramProcessor):
    matcher_type = NonSplittingChromatogramMatcher
