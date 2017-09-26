import time
from collections import defaultdict
from itertools import permutations

import numpy as np

from glycan_profiling.task import TaskBase

from glycan_profiling.chromatogram_tree import (
    Chromatogram, ChromatogramForest, Unmodified,
    mask_subsequence, DuplicateNodeError, get_chromatogram,
    SimpleChromatogram, find_truncation_points,
    ChromatogramFilter, GlycanCompositionChromatogram, GlycopeptideChromatogram,
    ChromatogramOverlapSmoother, ChromatogramGraph)

from .scan_cache import NullScanCacheHandler

from .scoring import (
    ChromatogramSolution,
    ChromatogramScorer)

from .composition_distribution_model import smooth_network, display_table


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


def prune_bad_adduct_branches(solutions, score_margin=0.05, ratio_threshold=1.5):
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
                is_close = abs(case.score - owner_item.score) < score_margin
                is_weaker = case.score > owner_item.score
                # If the owning group is lower scoring, but the scores are close
                if is_weaker and is_close:
                    component_signal = case.total_signal
                    complement_signal = owner_item.total_signal - component_signal
                    signal_ratio = complement_signal / component_signal
                    # The owner is more abundant than used-as-adduct-case
                    if signal_ratio > ratio_threshold:
                        is_weaker = False
                # If the scores are close, but the owning group is less abundant,
                # e.g. more mass shift groups or mass accuracy prevents propagation
                # of mass shifts
                elif is_close and (owner_item.total_signal / case.total_signal) < 1:
                    is_weaker = True
                if is_weaker:
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
    out = ChromatogramFilter(out)
    return out


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
                    name = match
                    if name in exclude_compositions:
                        # This chromatogram matches another form of an existing composition
                        # assignment. If it were assigned during `join_mass_shifted`, then
                        # it overlapped with that entity and should not be merged. Otherwise
                        # construct a new match
                        for hit in exclude_compositions[name]:
                            if span_overlap(hit, chroma):
                                exclude = True
                                break
                        else:
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
                    else:
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

    def join_common_identities(self, chromatograms, delta_rt=0):
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
                if last.overlaps_in_time(case) or ((case.start_time - last.end_time) < delta_rt):
                    merged = last._merge_missing_only(case)
                    merged.used_as_adduct = list(last.used_as_adduct)
                    for ua in case.used_as_adduct:
                        if ua not in merged.used_as_adduct:
                            merged.used_as_adduct.append(ua)
                    last = merged
                    last.created_at = "join_common_identities"
                else:
                    accumulated.append(last)
                    last = case
            accumulated.append(last)
            out.extend(accumulated)
        return ChromatogramFilter(out)

    def find_related_profiles(self, chromatograms, adducts, mass_error_tolerance=1e-5):
        graph = ChromatogramGraph(chromatograms)
        graph.find_shared_peaks()
        components = graph.connected_components()

        for component in components:
            component = [node.chromatogram for node in component]
            if len(component) == 1:
                continue
            problem_pairs = set()
            for a, b in permutations(component, 2):
                best_err = float('inf')
                best_match = None
                mass_shift = a.weighted_neutral_mass - b.weighted_neutral_mass
                if mass_shift != 0:
                    for adduct in adducts:
                        err = abs((adduct.mass - mass_shift) / mass_shift)
                        if err < mass_error_tolerance and err < best_err:
                            best_err = err
                            best_match = adduct
                else:
                    # self.log("%r and %r have a 0 mass shift." % (a, b))
                    problem_pairs.add(frozenset((a, b)))
                if best_match is None:
                    # these two chromatograms may be adducts already.
                    used_as_adduct = False
                    for key, shift_type in a.used_as_adduct:
                        if key == b.key:
                            used_as_adduct = True
                    if used_as_adduct:
                        continue
                    for key, shift_type in b.used_as_adduct:
                        if key == a.key:
                            used_as_adduct = True
                    if used_as_adduct:
                        continue
                    mass_diff_ppm = abs((a.theoretical_mass - b.theoretical_mass) /
                                        b.theoretical_mass)
                    if mass_diff_ppm < mass_error_tolerance:
                        # self.log(
                        #     ("There is a peak-sharing relationship between %r and %r"
                        #      " which may indicating these two entities should be"
                        #      " merged.") % (a, b))
                        pass
                    else:
                        # really ambiguous. needs more attention.
                        if frozenset((a, b)) in problem_pairs:
                            continue

                        # self.log(
                        #     ("There is a peak-sharing relationship between %r"
                        #      " and %r (%g) but no experimental mass shift could be"
                        #      " found to explain it") % (
                        #         a, b, mass_diff_ppm * b.theoretical_mass))
                        problem_pairs.add(frozenset((a, b)))
                else:
                    used_set = set(b.used_as_adduct)
                    used_set.add((a.key, best_match))
                    b.used_as_adduct = list(used_set)

    def process(self, chromatograms, adducts=None, mass_error_tolerance=1e-5, delta_rt=0):
        if adducts is None:
            adducts = []
        matches = []
        chromatograms = ChromatogramFilter(chromatograms)
        self.log("Matching chromatograms")
        i = 0
        n = len(chromatograms)
        for chro in chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("%0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            matches.extend(self.search(chro, mass_error_tolerance))
        matches = ChromatogramFilter(matches)
        matches = self.join_common_identities(matches, delta_rt)
        matches = self.join_mass_shifted(matches, adducts, mass_error_tolerance)
        if adducts:
            self.log("Handling Adducts")
            matches = self.reverse_adduct_search(matches, adducts, mass_error_tolerance)
        matches = self.join_common_identities(matches, delta_rt)
        self.find_related_profiles(matches, adducts, mass_error_tolerance)
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
    acceptance_threshold = 0.4
    ignore_below = 1e-5

    def __init__(self, scoring_model=None):
        if scoring_model is None:
            scoring_model = ChromatogramScorer()
        self.scoring_model = scoring_model

    def configure(self, analysis_info):
        self.scoring_model.configure(analysis_info)

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        filtered = ChromatogramFilter.process(
            chromatograms, delta_rt=delta_rt, min_points=min_points)
        if smooth_overlap_rt:
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
                sol = self.evaluate_chromatogram(case)
                if self.scoring_model.accept(sol):
                    solutions.append(sol)
                end = time.time()
                # Report on anything that took more than 30 seconds to evaluate
                if end - start > 30.0:
                    self.log("%r took a long time to evaluated (%0.2fs)" % (case, end - start))
            except (IndexError, ValueError):
                continue
        return ChromatogramFilter(solutions)

    def evaluate_chromatogram(self, chromatogram):
        score_set = self.scoring_model.compute_scores(chromatogram)
        score = score_set.product()
        return ChromatogramSolution(
            chromatogram, score, scorer=self.scoring_model,
            score_set=score_set)

    def finalize_matches(self, solutions):
        out = []
        for sol in solutions:
            if sol.score <= self.ignore_below:
                continue
            elif (sol.composition is None) and (Unmodified not in sol.adducts):
                continue
            out.append(sol)
        solutions = ChromatogramFilter(out)
        return solutions

    def score(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
              adducts=None, *args, **kwargs):

        solutions = self.evaluate(
            chromatograms, delta_rt, min_points, smooth_overlap_rt, *args, **kwargs)

        if adducts:
            hold = self.prune_adducts(solutions)
            self.log("Re-evaluating after adduct pruning")
            solutions = self.evaluate(hold, delta_rt, min_points, smooth_overlap_rt,
                                      *args, **kwargs)

        solutions = self.finalize_matches(solutions)
        return solutions

    def prune_adducts(self, solutions):
        return prune_bad_adduct_branches(ChromatogramFilter(solutions))

    def acceptance_filter(self, solutions, threshold=None):
        if threshold is None:
            threshold = self.acceptance_threshold
        return ChromatogramFilter([
            sol for sol in solutions
            if sol.score >= threshold and not sol.used_as_adduct
        ])

    def update_parameters(self, param_dict):
        param_dict['scoring_model'] = self.scoring_model


class LogitSumChromatogramEvaluator(ChromatogramEvaluator):
    acceptance_threshold = 4
    ignore_below = 2

    def __init__(self, scorer):
        super(LogitSumChromatogramEvaluator, self).__init__(scorer)

    def prune_adducts(self, solutions):
        return prune_bad_adduct_branches(ChromatogramFilter(solutions), score_margin=2.5)

    def evaluate_chromatogram(self, chromatogram):
        score_set = self.scoring_model.compute_scores(chromatogram)
        logitsum_score = score_set.logitsum()
        return ChromatogramSolution(
            chromatogram, logitsum_score, scorer=self.scoring_model,
            score_set=score_set)

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        solutions = super(LogitSumChromatogramEvaluator, self).evaluate(
            chromatograms, delta_rt=delta_rt, min_points=min_points,
            smooth_overlap_rt=smooth_overlap_rt, *args, **kwargs)

        accumulator = defaultdict(list)
        for case in solutions:
            accumulator[case.key].append(case)
        solutions = []
        for group, members in accumulator.items():
            members = sorted(members, key=lambda x: x.score, reverse=True)
            reference = members[0]
            base = reference.clone()
            for other in members[1:]:
                base = base.merge(other)
            merged = reference.__class__(
                base, reference.score, scorer=reference.scorer,
                score_set=reference.score_set)
            solutions.append(merged)
        return ChromatogramFilter(solutions)


class LaplacianRegularizedChromatogramEvaluator(LogitSumChromatogramEvaluator):
    def __init__(self, scorer, network, smoothing_factor=None, grid_smoothing_max=1.0,
                 regularization_model=None):
        super(LaplacianRegularizedChromatogramEvaluator,
              self).__init__(scorer)
        self.network = network
        self.smoothing_factor = smoothing_factor
        self.grid_smoothing_max = grid_smoothing_max
        self.regularization_model = regularization_model

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        solutions = super(LaplacianRegularizedChromatogramEvaluator, self).evaluate(
            chromatograms, delta_rt=delta_rt, min_points=min_points,
            smooth_overlap_rt=smooth_overlap_rt, *args, **kwargs)
        self.log("... Applying Network Smoothing Regularization")
        updated_network, search, params = smooth_network(
            self.network, solutions, lmbda=self.smoothing_factor,
            lambda_max=self.grid_smoothing_max,
            model_state=self.regularization_model)
        solutions = sorted(solutions, key=lambda x: x.score, reverse=True)
        # TODO - Use aggregation across multiple observations for the same glycan composition
        # instead of discarding all but the top scoring feature?
        seen = dict()
        unannotated = []
        for sol in solutions:
            if sol.glycan_composition is None:
                unannotated.append(sol)
                continue
            if sol.glycan_composition in seen:
                continue
            seen[sol.glycan_composition] = sol
            node = updated_network[sol.glycan_composition]
            if sol.score > self.acceptance_threshold:
                sol.score = node.score
            else:
                # Do not permit network smoothing to boost scores below acceptance_threshold
                if node.score < sol.score:
                    sol.score = node.score
        self.network_parameters = params
        self.grid_search = search
        display_table(
            search.model.neighborhood_names,
            np.array(params.tau).reshape((-1, 1)),
            print_fn=lambda x: self.log("...... %s" % (x,)))
        self.log("...... smoothing factor: %0.3f; threshold: %0.3f" % (
            params.lmbda, params.threshold))
        return ChromatogramFilter(list(seen.values()) + unannotated)

    def update_parameters(self, param_dict):
        super(LaplacianRegularizedChromatogramEvaluator, self).update_parameters(param_dict)
        param_dict['network_parameters'] = self.network_parameters
        param_dict['network_model'] = self.grid_search


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
        self.accumulated = self.peak_loader.ms1_peaks_above(self.minimum_mass, self.minimum_intensity)
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
                 scoring_model=None, smooth_overlap_rt=True, acceptance_threshold=0.4,
                 delta_rt=0.25, peak_loader=None):
        if adducts is None:
            adducts = []
        self._chromatograms = chromatograms
        self.database = database
        self.adducts = adducts
        self.mass_error_tolerance = mass_error_tolerance
        self.peak_loader = peak_loader

        self.scoring_model = scoring_model

        self.smooth_overlap_rt = smooth_overlap_rt
        self.acceptance_threshold = acceptance_threshold
        self.delta_rt = delta_rt

        self.solutions = None
        self.accepted_solutions = None

    def make_matcher(self):
        matcher = self.matcher_type(self.database)
        return matcher

    def _match_compositions(self):
        matcher = self.make_matcher()
        matches = matcher.process(
            self._chromatograms, self.adducts, self.mass_error_tolerance,
            delta_rt=(self.delta_rt * 4 if self.smooth_overlap_rt else 0))
        return matches

    def make_evaluator(self):
        evaluator = ChromatogramEvaluator(self.scoring_model)
        return evaluator

    def match_compositions(self):
        self.log("Begin Matching Chromatograms")
        matches = self._match_compositions()
        self.log("End Matching Chromatograms")
        self.log("%d Chromatogram Candidates Found" % (len(matches),))
        return matches

    def evaluate_chromatograms(self, matches):
        self.log("Begin Evaluating Chromatograms")
        self.evaluator = self.make_evaluator()
        self.evaluator.configure({
            "peak_loader": self.peak_loader,
            "adducts": self.adducts,
            "delta_rt": self.delta_rt,
            "mass_error_tolerance": self.mass_error_tolerance,
        })
        self.solutions = self.evaluator.score(
            matches, smooth_overlap_rt=self.smooth_overlap_rt,
            adducts=self.adducts, delta_rt=self.delta_rt)
        self.accepted_solutions = self.evaluator.acceptance_filter(self.solutions)
        self.log("End Evaluating Chromatograms")

    def run(self):
        matches = self.match_compositions()
        self.evaluate_chromatograms(matches)

    def __iter__(self):
        if self.accepted_solutions is None:
            self.run()
        return iter(self.accepted_solutions)


class LogitSumChromatogramProcessor(ChromatogramProcessor):
    def make_evaluator(self):
        evaluator = LogitSumChromatogramEvaluator(self.scoring_model)
        return evaluator


class LaplacianRegularizedChromatogramProcessor(LogitSumChromatogramProcessor):
    GRID_SEARCH = 'grid'

    def __init__(self, chromatograms, database, adducts=None, mass_error_tolerance=1e-5,
                 scoring_model=None, smooth_overlap_rt=True, acceptance_threshold=0.4,
                 delta_rt=0.25, peak_loader=None, smoothing_factor=0.2, grid_smoothing_max=1.0,
                 regularization_model=None):
        super(LaplacianRegularizedChromatogramProcessor, self).__init__(
            chromatograms, database, adducts, mass_error_tolerance,
            scoring_model, smooth_overlap_rt, acceptance_threshold,
            delta_rt, peak_loader)
        if grid_smoothing_max is None:
            grid_smoothing_max = 1.0
        if self.GRID_SEARCH == smoothing_factor:
            smoothing_factor = None
        self.smoothing_factor = smoothing_factor
        self.grid_smoothing_max = grid_smoothing_max
        self.regularization_model = regularization_model

    def make_evaluator(self):
        evaluator = LaplacianRegularizedChromatogramEvaluator(
            self.scoring_model,
            self.database.glycan_composition_network,
            smoothing_factor=self.smoothing_factor,
            grid_smoothing_max=self.grid_smoothing_max,
            regularization_model=self.regularization_model)
        return evaluator


class GlycopeptideChromatogramProcessor(ChromatogramProcessor):
    matcher_type = GlycopeptideChromatogramMatcher


class NonSplittingChromatogramProcessor(ChromatogramProcessor):
    matcher_type = NonSplittingChromatogramMatcher
