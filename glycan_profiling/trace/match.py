from itertools import permutations
from collections import defaultdict

from glycan_profiling.chromatogram_tree import (
    Chromatogram,
    Unmodified,
    DuplicateNodeError,
    ChromatogramFilter,
    GlycanCompositionChromatogram,
    GlycopeptideChromatogram,
    ChromatogramGraph)

from glycan_profiling.task import TaskBase


def span_overlap(reference, target):
    """test whether two time Chromatogram objects
    overlap each other in the time domain

    Parameters
    ----------
    reference: Chromatogram
    target: Chromatogram

    Returns
    -------
    bool
    """
    cond = ((reference.start_time <= target.start_time and reference.end_time >= target.end_time) or (
        reference.start_time >= target.start_time and reference.end_time <= target.end_time) or (
        reference.start_time >= target.start_time and reference.end_time >= target.end_time and
        reference.start_time <= target.end_time) or (
        reference.start_time <= target.start_time and reference.end_time >= target.start_time) or (
        reference.start_time <= target.end_time and reference.end_time >= target.end_time))
    return cond


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
                for m in sorted(matches, key=lambda x: getattr(x, "calculated_mass", 0))])
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
        n = len(chromatograms)
        i = 0
        self.log("Begin Reverse Search")
        for chroma in candidate_chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("... %0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
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
        n = len(chromatograms)
        self.log("Begin Forward Search")
        for chroma in chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("... %0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            add = chroma
            for adduct in adducts:
                query_mass = chroma.weighted_neutral_mass + adduct.mass
                matches = chromatograms.find_all_by_mass(query_mass, mass_error_tolerance)
                for match in matches:
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

    def search_all(self, chromatograms, mass_error_tolerance=1e-5):
        matches = []
        chromatograms = ChromatogramFilter(chromatograms)
        self.log("Matching Chromatograms")
        i = 0
        n = len(chromatograms)
        for chro in chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("... %0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            matches.extend(self.search(chro, mass_error_tolerance))
        matches = ChromatogramFilter(matches)
        return matches

    def process(self, chromatograms, adducts=None, mass_error_tolerance=1e-5, delta_rt=0):
        if adducts is None:
            adducts = []
        matches = []
        chromatograms = ChromatogramFilter(chromatograms)
        matches = self.search_all(chromatograms, mass_error_tolerance)
        matches = self.join_common_identities(matches, delta_rt)
        if adducts:
            self.log("Handling Adducts")
            matches = self.join_mass_shifted(matches, adducts, mass_error_tolerance)
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
        super(GlycopeptideChromatogramMatcher, self).__init__(database, chromatogram_type)


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
