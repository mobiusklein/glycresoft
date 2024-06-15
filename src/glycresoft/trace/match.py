from collections import defaultdict

from glycresoft.database.mass_collection import NeutralMassDatabase

from glycresoft.chromatogram_tree import (
    Chromatogram,
    Unmodified,
    DuplicateNodeError,
    ChromatogramFilter,
    GlycanCompositionChromatogram,
    GlycopeptideChromatogram,
    ChromatogramGraph)

from glycresoft.task import TaskBase


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
    memory_load_threshold = 1e5

    def __init__(self, database, chromatogram_type=None, require_unmodified=False, enforce_charges=False):
        if chromatogram_type is None:
            chromatogram_type = GlycanCompositionChromatogram
        self.database = database
        self.require_unmodified = require_unmodified
        self.enforce_charges = enforce_charges
        self._group_bundle = dict()
        self.chromatogram_type = chromatogram_type
        if len(database) < self.memory_load_threshold:
            self._in_memory = True
            self.database = NeutralMassDatabase(list(database.get_all_records()))
            self._original_convert = database._convert
        else:
            self._in_memory = False

    def _convert(self, record):
        if not self._in_memory:
            return self.database._convert(record)
        else:
            return self._original_convert(record)

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
                self._convert(m)
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

    def reverse_mass_shift_search(self, chromatograms, mass_shifts, mass_error_tolerance=1e-5):
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
            for mass_shift in mass_shifts:
                matches = self.match(candidate_mass - mass_shift.mass, mass_error_tolerance)
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
                                chroma_to_update.created_at = "reverse_mass_shiftion_search"
                            chroma, _ = chroma.bisect_mass_shift(Unmodified)
                            chroma_to_update = chroma_to_update.merge(chroma, mass_shift)
                            chroma_to_update.created_at = "reverse_mass_shiftion_search"
                            new_members[name] = chroma_to_update
                            matched = True
                    else:
                        if name in new_members:
                            chroma_to_update = new_members[name]
                        else:
                            chroma_to_update = self.chromatogram_type(match)
                            chroma_to_update.created_at = "reverse_mass_shiftion_search"
                        chroma, _ = chroma.bisect_mass_shift(Unmodified)
                        chroma_to_update = chroma_to_update.merge(chroma, mass_shift)
                        chroma_to_update.created_at = "reverse_mass_shiftion_search"
                        new_members[name] = chroma_to_update
                        matched = True
            if not matched and not exclude:
                unmatched.append(chroma)
        out = []
        out.extend(s for g in exclude_compositions.values() for s in g)
        out.extend(new_members.values())
        out.extend(unmatched)
        return ChromatogramFilter(out)

    def join_mass_shifted(self, chromatograms, mass_shifts, mass_error_tolerance=1e-5):
        out = []
        i = 0
        n = len(chromatograms)
        self.log("Begin Forward Search")
        for chroma in chromatograms:
            i += 1
            if i % 1000 == 0:
                self.log("... %0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            add = chroma
            for mass_shift in mass_shifts:
                query_mass = chroma.weighted_neutral_mass + mass_shift.mass
                matches = chromatograms.find_all_by_mass(query_mass, mass_error_tolerance)
                for match in matches:
                    if match and span_overlap(add, match):
                        try:
                            match.used_as_mass_shift.append((add.key, mass_shift))
                            add = add.merge(match, node_type=mass_shift, skip_duplicate_nodes=True)
                            add.created_at = "join_mass_shifted"
                            add.mass_shifts.append(mass_shift)
                        except DuplicateNodeError as e:
                            e.original = chroma
                            e.to_add = match
                            e.accumulated = add
                            e.mass_shift = mass_shift
                            raise e
            out.append(add)
        return ChromatogramFilter(out)

    def join_common_identities(self, chromatograms, delta_rt=0):
        chromatograms._build_key_map()
        key_map = chromatograms._key_map
        out = []
        for _, disjoint_set in key_map.items():
            if len(tuple(disjoint_set)) == 1:
                out.extend(disjoint_set)
                continue

            accumulated = []
            last = disjoint_set[0]
            for case in disjoint_set[1:]:
                if last.overlaps_in_time(case) or ((case.start_time - last.end_time) < delta_rt):
                    merged = last._merge_missing_only(case)
                    merged.used_as_mass_shift = list(last.used_as_mass_shift)
                    for ua in case.used_as_mass_shift:
                        if ua not in merged.used_as_mass_shift:
                            merged.used_as_mass_shift.append(ua)
                    last = merged
                    last.created_at = "join_common_identities"
                else:
                    accumulated.append(last)
                    last = case
            accumulated.append(last)
            out.extend(accumulated)
        return ChromatogramFilter(out)

    def find_related_profiles(self, chromatograms, mass_shifts, mass_error_tolerance=1e-5):
        self.log("Building Connected Components")
        graph = ChromatogramGraph(chromatograms)
        graph.find_shared_peaks()
        components = graph.connected_components()

        n_components = len(components)
        self.log("Validating %d Components" % (n_components, ))
        for i_components, component in enumerate(components):
            if i_components % 1000 == 0 and i_components > 0:
                self.log("... %d Components Validated (%0.2f%%)" % (
                    i_components,
                    i_components / float(n_components) * 100.))
            if len(component) == 1:
                continue
            component = ChromatogramFilter([node.chromatogram for node in component])

            for a in component:
                pairs = []
                for mass_shift in mass_shifts:
                    bs = component.find_all_by_mass(
                        a.weighted_neutral_mass - mass_shift.mass, mass_error_tolerance)
                    for b in bs:
                        if b != a:
                            pairs.append((mass_shift, b))
                if not pairs:
                    continue
                grouped_pairs = []
                pairs.sort(key=lambda x: (x[1].start_time, x[1].weighted_neutral_mass))
                last = [pairs[0]]
                for current in pairs[1:]:
                    if current[1] is last[0][1]:
                        last.append(current)
                    else:
                        grouped_pairs.append(last)
                        last = [current]
                grouped_pairs.append(last)
                unique_pairs = []

                def minimizer(args):
                    mass_shift, b = args
                    return abs(a.weighted_neutral_mass - (b.weighted_neutral_mass + mass_shift.mass))

                for pair_group in grouped_pairs:
                    unique_pairs.append(min(pair_group, key=minimizer))

                for mass_shift, b in unique_pairs:
                    used_set = set(b.used_as_mass_shift)
                    used_set.add((a.key, mass_shift))
                    b.used_as_mass_shift = list(used_set)

    def search_all(self, chromatograms, mass_error_tolerance=1e-5):
        matches = []
        chromatograms = ChromatogramFilter(chromatograms)
        self.log("Matching Chromatograms")
        i = 0
        n = len(chromatograms)
        for chro in chromatograms:
            i += 1
            if i % 1000 == 0 and i > 0:
                self.log("... %0.2f%% chromatograms searched (%d/%d)" % (i * 100. / n, i, n))
            matches.extend(self.search(chro, mass_error_tolerance))
        matches = ChromatogramFilter(matches)
        return matches

    def strip_only_modified(self, matches):
        out = []
        for match in matches:
            adducts = match.adducts
            if Unmodified in adducts:
                out.append(match)
        return out

    def enforce_charge_counts(self, matches):
        out = []
        for match in matches:
            changed = False
            for adduct in match.adducts:
                if adduct.charge_carrier != 0:
                    charge_carrier = adduct.charge_carrier
                    charges = match.charge_states
                    has_sufficient_charge = [c for c in charges if c >= charge_carrier]
                    has_insufficient_charge = [c for c in charges if c < charge_carrier]
                    if has_insufficient_charge:
                        # TODO: Bisect the chromatogram along the relevant charge states
                        # and then extract out the disallowed adduction state, then re-combine
                        # the chromatogram.
                        pass
            if not changed:
                out.append(match)
        return out

    def process(self, chromatograms, mass_shifts=None, mass_error_tolerance=1e-5, delta_rt=0):
        if mass_shifts is None:
            mass_shifts = []
        matches = []
        chromatograms = ChromatogramFilter(chromatograms)
        matches = self.search_all(chromatograms, mass_error_tolerance)
        matches = self.join_common_identities(matches, delta_rt)
        if mass_shifts:
            self.log("Handling Mass Shifts")
            matches = self.join_mass_shifted(matches, mass_shifts, mass_error_tolerance)
            matches = self.reverse_mass_shift_search(matches, mass_shifts, mass_error_tolerance)
        matches = self.join_common_identities(matches, delta_rt)
        self.find_related_profiles(matches, mass_shifts, mass_error_tolerance)
        if self.require_unmodified:
            matches = self.strip_only_modified(matches)
        return matches


class GlycanChromatogramMatcher(ChromatogramMatcher):
    pass


class GlycopeptideChromatogramMatcher(ChromatogramMatcher):
    def __init__(self, database, chromatogram_type=None, require_unmodified=False, enforce_charges=False):
        if chromatogram_type is None:
            chromatogram_type = GlycopeptideChromatogram
        super(GlycopeptideChromatogramMatcher, self).__init__(
            database, chromatogram_type, require_unmodified=require_unmodified,
            enforce_charges=enforce_charges)


class NonSplittingChromatogramMatcher(ChromatogramMatcher):
    def __init__(self, database, chromatogram_type=None, require_unmodified=False, enforce_charges=False):
        if chromatogram_type is None:
            chromatogram_type = Chromatogram
        super(NonSplittingChromatogramMatcher, self).__init__(
            database, chromatogram_type, require_unmodified=require_unmodified,
            enforce_charges=enforce_charges)

    def assign(self, chromatogram, group):
        out = []
        if group is None:
            return [chromatogram]
        else:
            case = chromatogram.clone(self.chromatogram_type)
            case.composition = group
            out.append(case)
            return out
