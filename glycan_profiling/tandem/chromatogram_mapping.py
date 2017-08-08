from glycan_profiling.task import TaskBase, log_handle
from glycan_profiling.chromatogram_tree import ChromatogramWrapper, build_rt_interval_tree, ChromatogramFilter
from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from collections import defaultdict, namedtuple

SolutionEntry = namedtuple("SolutionEntry", "solution, score, percentile, best_score")


class SpectrumMatchSolutionCollectionBase(object):
    def _compute_representative_weights(self, threshold_fn=lambda x: True):
        scores = defaultdict(float)
        best_scores = defaultdict(float)
        for psm in self.tandem_solutions:
            if threshold_fn(psm):
                for sol in psm.get_top_solutions():
                    scores[sol.target] += (sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
        total = sum(scores.values())
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k]) for k, v in scores.items()
        ]
        weights.sort(key=lambda x: x.percentile, reverse=True)
        return weights

    def most_representative_solutions(self, threshold_fn=lambda x: True):
        weights = self._compute_representative_weights(threshold_fn)
        if weights:
            return [x for x in weights if abs(x.percentile - weights[0].percentile) < 1e-5]


class TandemAnnotatedChromatogram(ChromatogramWrapper, SpectrumMatchSolutionCollectionBase):
    def __init__(self, chromatogram):
        super(TandemAnnotatedChromatogram, self).__init__(chromatogram)
        self.tandem_solutions = []
        self.time_displaced_assignments = []
        self.best_msms_score = None

    def add_solution(self, item):
        case_mass = item.precursor_information.neutral_mass
        if abs(case_mass - self.chromatogram.neutral_mass) > 100:
            log_handle.log("Warning, mis-assigned spectrum match to chromatogram %r, %r" % (self, item))
        self.tandem_solutions.append(item)

    def add_displaced_solution(self, item):
        self.add_solution(item)

    def clone(self):
        new = super(TandemAnnotatedChromatogram, self).clone()
        new.tandem_solutions = list(self.tandem_solutions)
        new.time_displaced_assignments = list(self.time_displaced_assignments)
        new.best_msms_score = self.best_msms_score

    def merge(self, other):
        new = self.__class__(self.chromatogram.merge(other.chromatogram))
        new.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        new.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments
        return new

    def merge_in_place(self, other):
        new = self.chromatogram.merge(other.chromatogram)
        self.chromatogram = new
        self.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        self.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments

    def assign_entity(self, solution_entry, entity_chromatogram_type=GlycopeptideChromatogram):
        entity_chroma = entity_chromatogram_type(
            self.chromatogram.composition,
            self.chromatogram.nodes, self.chromatogram.adducts,
            self.chromatogram.used_as_adduct)
        entity_chroma.entity = solution_entry.solution
        self.chromatogram = entity_chroma
        self.best_msms_score = solution_entry.best_score


class TandemSolutionsWithoutChromatogram(SpectrumMatchSolutionCollectionBase):
    def __init__(self, entity, tandem_solutions):
        self.entity = entity
        self.composition = entity
        self.tandem_solutions = tandem_solutions

    @classmethod
    def aggregate(cls, solutions):
        collect = defaultdict(list)
        for solution in solutions:
            best_match = solution.best_solution()
            collect[best_match.target.id].append(solution)
        out = []
        for group in collect.values():
            solution = group[0]
            best_match = solution.best_solution()
            structure = best_match.target
            out.append(cls(structure, group))
        return out


class ScanTimeBundle(object):
    def __init__(self, solution, scan_time):
        self.solution = solution
        self.scan_time = scan_time

    @property
    def score(self):
        try:
            return self.solution.score
        except AttributeError:
            return None

    def __hash__(self):
        return hash((self.solution, self.scan_time))

    def __eq__(self, other):
        return self.solution == other.solution and self.scan_time == other.scan_time

    def __repr__(self):
        return "ScanTimeBundle(%s, %0.4f, %0.4f)" % (
            self.solution.scan.id, self.score, self.scan_time)


def aggregate_by_assigned_entity(annotated_chromatograms, delta_rt=0.25):
    aggregated = defaultdict(list)
    finished = []
    for chroma in annotated_chromatograms:
        if chroma.composition is not None:
            if chroma.entity is not None:
                aggregated[chroma.entity].append(chroma)
            else:
                aggregated[chroma.composition].append(chroma)
        else:
            finished.append(chroma)
    for entity, group in aggregated.items():
        out = []
        group = sorted(group, key=lambda x: x.start_time)
        chroma = group[0]
        for obs in group[1:]:
            if chroma.chromatogram.overlaps_in_time(obs) or (
                    chroma.end_time - obs.start_time) < delta_rt:
                chroma = chroma.merge(obs)
            else:
                out.append(chroma)
                chroma = obs
        out.append(chroma)
        finished.extend(out)
    return finished


class ChromatogramMSMSMapper(TaskBase):
    def __init__(self, chromatograms, error_tolerance=1e-5, scan_id_to_rt=lambda x: x):
        self.chromatograms = ChromatogramFilter(map(
            TandemAnnotatedChromatogram, chromatograms))
        self.rt_tree = build_rt_interval_tree(self.chromatograms)
        self.scan_id_to_rt = scan_id_to_rt
        self.orphans = []
        self.error_tolerance = error_tolerance

    def find_chromatogram_spanning(self, time):
        return ChromatogramFilter([interv[0] for interv in self.rt_tree.contains_point(time)])

    def find_chromatogram_for(self, solution):
        precursor_scan_time = self.scan_id_to_rt(
            solution.precursor_information.precursor_scan_id)
        overlapping_chroma = self.find_chromatogram_spanning(precursor_scan_time)
        chroma = overlapping_chroma.find_mass(
            solution.precursor_information.neutral_mass, self.error_tolerance)
        if chroma is None:
            self.orphans.append(ScanTimeBundle(solution, precursor_scan_time))
        else:
            chroma.tandem_solutions.append(solution)

    def assign_solutions_to_chromatograms(self, solutions):
        for solution in solutions:
            self.find_chromatogram_for(solution)

    def distribute_orphans(self, threshold_fn=lambda x: x.q_value < 0.05):
        lost = []
        for orphan in self.orphans:
            mass = orphan.solution.precursor_ion_mass
            window = self.error_tolerance * mass
            candidates = self.chromatograms.mass_between(mass - window, mass + window)
            time = orphan.scan_time
            if len(candidates) > 0:
                best_index = 0
                best_distance = float('inf')
                for i, candidate in enumerate(candidates):
                    dist = min(abs(candidate.start_time - time), abs(candidate.end_time - time))
                    if dist < best_distance:
                        best_index = i
                        best_distance = dist
                    new_owner = candidates[best_index]
                    new_owner.add_displaced_solution(orphan.solution)
            else:
                if threshold_fn(orphan.solution):
                    self.log("No chromatogram found for %r, q-value %0.4f (mass: %0.4f, time: %0.4f)" % (
                        orphan, orphan.solution.q_value, mass, time))
                    lost.append(orphan.solution)
        self.orphans = TandemSolutionsWithoutChromatogram.aggregate(lost)

    def assign_entities(self, threshold_fn=lambda x: x.q_value < 0.05, entity_chromatogram_type=None):
        if entity_chromatogram_type is None:
            entity_chromatogram_type = GlycopeptideChromatogram
        for chromatogram in self:
            solutions = chromatogram.most_representative_solutions(threshold_fn)
            if solutions:
                solutions = sorted(solutions, key=lambda x: x.score, reverse=True)
                chromatogram.assign_entity(solutions[0], entity_chromatogram_type=entity_chromatogram_type)

    def merge_common_entities(self, annotated_chromatograms):
        aggregated = defaultdict(list)
        finished = []
        for chroma in annotated_chromatograms:
            if chroma.composition is not None:
                if chroma.entity is not None:
                    aggregated[chroma.entity].append(chroma)
                else:
                    aggregated[chroma.composition].append(chroma)
            else:
                finished.append(chroma)
        for entity, group in aggregated.items():
            out = []
            chroma = group[0]
            for obs in group[1:]:
                if chroma.chromatogram.overlaps_in_time(obs):
                    chroma = chroma.merge(obs)
                else:
                    out.append(chroma)
                    chroma = obs
            out.append(chroma)
            finished.extend(out)
        return finished

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]
