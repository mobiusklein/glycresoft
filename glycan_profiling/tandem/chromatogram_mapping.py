import os
from glycan_profiling.task import TaskBase, log_handle
from glycan_profiling.chromatogram_tree import (
    ChromatogramWrapper, build_rt_interval_tree, ChromatogramFilter,
    Unmodified)

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from .spectrum_match.solution_set import NOParsimonyMixin

from collections import defaultdict, namedtuple

SolutionEntry = namedtuple("SolutionEntry", "solution, score, percentile, best_score, match")

debug_mode = bool(os.environ.get('GLYCRESOFTDEBUG', False))


class NOParsimonyRepresentativeSelector(NOParsimonyMixin):
    def get_score(self, solution):
        return solution.percentile

    def get_target(self, solution):
        return solution.match.target

    def sort(self, solution_set):
        solution_set = sorted(solution_set, key=lambda x: x.percentile, reverse=True)
        try:
            if solution_set and self.get_target(solution_set[0]).is_o_glycosylated():
                solution_set = self.hoist_equivalent_n_linked_solution(solution_set)
        except AttributeError:
            import warnings
            warnings.warn("Could not determine glycosylation state of target of type %r" % type(self.get_target(solution_set[0])))
        return solution_set

    def __call__(self, solution_set):
        return self.sort(solution_set)


parsimony_sort = NOParsimonyRepresentativeSelector()


class SpectrumMatchSolutionCollectionBase(object):
    def _compute_representative_weights(self, threshold_fn=lambda x: True, reject_shifted=False):
        """Calculate a total score for all matched structures across all time points for this
        solution collection, and rank them.

        This total score is the sum of the score over all spectrum matches for which that
        structure was the best match. The percentile is the ratio of the total score for the
        ith structure divided by the sum of total scores over all structures.

        Parameters
        ----------
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False

        Returns
        -------
        list
            A list of solutions, ranked by percentile.
        """
        scores = defaultdict(float)
        best_scores = defaultdict(float)
        best_spectrum_match = dict()
        for psm in self.tandem_solutions:
            if threshold_fn(psm):
                for sol in psm.get_top_solutions():
                    if not threshold_fn(sol):
                        continue
                    if reject_shifted and sol.mass_shift != Unmodified:
                        continue
                    scores[sol.target] += (sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
                        best_spectrum_match[sol.target] = sol
        total = sum(scores.values())
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k],
                          best_spectrum_match[k]) for k, v in scores.items()
            if k in best_spectrum_match
        ]
        weights = parsimony_sort(weights)
        return weights

    def most_representative_solutions(self, threshold_fn=lambda x: True, reject_shifted=False):
        """Find the most representative solutions, the (very nearly the same, hopefully) structures with
        the highest aggregated score across all MSn events assigned to this collection.

        Parameters
        ----------
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False

        Returns
        -------
        list
            A list of solutions with approximately the greatest weight
        """
        weights = self._compute_representative_weights(threshold_fn, reject_shifted)
        if weights:
            representers = [x for x in weights if (weights[0].percentile - x.percentile) < 1e-5]
            return representers
        else:
            return []


class TandemAnnotatedChromatogram(ChromatogramWrapper, SpectrumMatchSolutionCollectionBase):
    def __init__(self, chromatogram):
        super(TandemAnnotatedChromatogram, self).__init__(chromatogram)
        self.tandem_solutions = []
        self.time_displaced_assignments = []
        self.best_msms_score = None
        self.representative_solutions = None

    def bisect_charge(self, charge):
        new_charge, new_no_charge = map(self.__class__, self.chromatogram.bisect_charge(charge))
        for hit in self.tandem_solutions:
            if hit.precursor_information.charge == charge:
                new_charge.add_solution(hit)
            else:
                new_no_charge.add_solution(hit)
        return new_charge, new_no_charge

    def bisect_mass_shift(self, mass_shift):
        new_mass_shift, new_no_mass_shift = map(self.__class__, self.chromatogram.bisect_mass_shift(mass_shift))
        for hit in self.tandem_solutions:
            if hit.best_solution().mass_shift == mass_shift:
                new_mass_shift.add_solution(hit)
            else:
                new_no_mass_shift.add_solution(hit)
        return new_mass_shift, new_no_mass_shift

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
        return new

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
            None,
            self.chromatogram.nodes, self.chromatogram.mass_shifts,
            self.chromatogram.used_as_mass_shift)
        entity_chroma.entity = solution_entry.solution
        if solution_entry.match.mass_shift != Unmodified:
            identified_shift = solution_entry.match.mass_shift
            for node in entity_chroma.nodes.unspool():
                if node.node_type == Unmodified:
                    node.node_type = identified_shift
                else:
                    node.node_type = (node.node_type + identified_shift)
            entity_chroma.invalidate()
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



class AnnotatedChromatogramAggregator(TaskBase):
    def __init__(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                threshold_fn=lambda x: x.q_value < 0.05):
        self.annotated_chromatograms = annotated_chromatograms
        self.delta_rt = delta_rt
        self.require_unmodified = require_unmodified
        self.threshold_fn = threshold_fn

    def combine_chromatograms(self, aggregated):
        merged = []
        for entity, group in aggregated.items():
            out = []
            group = sorted(group, key=lambda x: x.start_time)
            chroma = group[0]
            for obs in group[1:]:
                if chroma.chromatogram.overlaps_in_time(obs) or (
                        chroma.end_time - obs.start_time) < self.delta_rt:
                    chroma = chroma.merge(obs)
                else:
                    out.append(chroma)
                    chroma = obs
            out.append(chroma)
            merged.extend(out)
        return merged

    def aggregate_by_annotation(self, annotated_chromatograms):
        finished = []
        aggregated = defaultdict(list)
        for chroma in annotated_chromatograms:
            if chroma.composition is not None:
                if chroma.entity is not None:
                    # Convert to string to avoid redundant sequences from getting
                    # binned differently due to random ordering of ids.
                    aggregated[str(chroma.entity)].append(chroma)
                else:
                    aggregated[str(chroma.composition)].append(chroma)
            else:
                finished.append(chroma)
        return finished, aggregated

    def aggregate(self, annotated_chromatograms):
        self.log("Aggregating Common Entities: %d chromatograms" % (len(annotated_chromatograms,)))
        finished, aggregated = self.aggregate_by_annotation(annotated_chromatograms)
        finished.extend(self.combine_chromatograms(aggregated))
        self.log("After Merging: %d chromatograms" % (len(finished),))
        return finished

    def replace_solution(self, chromatogram, solutions):
        # select the best solution
        solutions = sorted(
                solutions, key=lambda x: x.score, reverse=True)

        # remove the invalidated mass shifts so that we're dealing with the raw masses again.
        current_shifts = chromatogram.chromatogram.mass_shifts
        if len(current_shifts) > 1:
            self.log("... Found a multiply shifted identification with no Unmodified state: %s" % (chromatogram.entity, ))
        partitions = []
        for shift in current_shifts:
            # The _ collects the "not modified with shift" portion of the chromatogram
            # we'll strip out in successive iterations. By virtue of reaching this point,
            # we're never dealing with an Unmodified portion.
            partition, _ = chromatogram.chromatogram.bisect_mass_shift(shift)
            # If we're somehow dealing with a compound mass shift here, remove
            # the bad shift from the composite, and reset the modified nodes to
            # Unmodified.
            partitions.append(partition.deduct_node_type(shift))
        # Merge in
        accumulated_chromatogram = partitions[0]
        for partition in partitions[1:]:
            accumulated_chromatogram = accumulated_chromatogram.merge(partition)
        chromatogram.chromatogram = accumulated_chromatogram

        # update the tandem annotations
        chromatogram.assign_entity(
            solutions[0],
            entity_chromatogram_type=chromatogram.chromatogram.__class__)
        chromatogram.representative_solutions = solutions
        return chromatogram

    def reassign_modified_only_cases(self, annotated_chromatograms):
        out = []
        for chromatogram in annotated_chromatograms:
            # the structure's best match has not been identified in an unmodified state
            if Unmodified not in chromatogram.mass_shifts:
                original_entity = getattr(chromatogram, "entity")
                solutions = chromatogram.most_representative_solutions(
                    self.threshold_fn, reject_shifted=True)
                # if there is a reasonable solution in an unmodified state
                if solutions:
                    chromatogram = self.replace_solution(chromatogram, solutions)
                    self.debug("... Replacing %s with %s", original_entity, chromatogram.entity)
                    out.append(chromatogram)
                else:
                    self.log(
                        "... Could not find an alternative option for %r" % (chromatogram,))
                    out.append(chromatogram)
            else:
                out.append(chromatogram)
        return out

    def run(self):
        merged_chromatograms = self.aggregate(self.annotated_chromatograms)
        if self.require_unmodified:
            spliced = self.reassign_modified_only_cases(merged_chromatograms)
            merged_chromatograms = self.aggregate(spliced)
        return ChromatogramFilter(merged_chromatograms)


def aggregate_by_assigned_entity(annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                                 threshold_fn=lambda x: x.q_value < 0.05):
    job = AnnotatedChromatogramAggregator(
        annotated_chromatograms, delta_rt=delta_rt,
        require_unmodified=require_unmodified, threshold_fn=threshold_fn)
    finished = job.run()
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
        try:
            precursor_scan_time = self.scan_id_to_rt(
                solution.precursor_information.precursor_scan_id)
        except Exception:
            precursor_scan_time = self.scan_id_to_rt(solution.scan_id)
        overlapping_chroma = self.find_chromatogram_spanning(precursor_scan_time)
        chroma = overlapping_chroma.find_mass(
            solution.precursor_information.neutral_mass, self.error_tolerance)
        if chroma is None:
            if debug_mode:
                self.log("... %s is an orphan" % (solution, ))
            self.orphans.append(ScanTimeBundle(solution, precursor_scan_time))
        else:
            if debug_mode:
                self.log("... Assigning %s to %s" % (solution, chroma))
            chroma.tandem_solutions.append(solution)

    def assign_solutions_to_chromatograms(self, solutions):
        n = len(solutions)
        for i, solution in enumerate(solutions):
            if i % 1000 == 0:
                self.log("... %d/%d Solutions Handled (%0.2f%%)" % (i, n, (i * 100.0 / n)))
            self.find_chromatogram_for(solution)

    def distribute_orphans(self, threshold_fn=lambda x: x.q_value < 0.05):
        lost = []
        n = len(self.orphans)
        n_chromatograms = len(self.chromatograms)
        for j, orphan in enumerate(self.orphans):
            mass = orphan.solution.precursor_ion_mass
            time = orphan.scan_time
            if j % 100 == 0:
                self.log("... %r %d/%d Orphans Handled (%0.2f%%)" % (orphan, j, n, (j * 100.0 / n)))
            candidates = self.chromatograms.find_all_by_mass(mass, self.error_tolerance)
            if len(candidates) > 0:
                best_index = 0
                best_distance = float('inf')
                for i, candidate in enumerate(candidates):
                    dist = min(abs(candidate.start_time - time), abs(candidate.end_time - time))
                    if dist < best_distance:
                        best_index = i
                        best_distance = dist
                new_owner = candidates[best_index]
                if debug_mode:
                    self.log("... Assigning %r to %r with %d existing solutions with distance %0.3f" %
                             (orphan, new_owner, len(new_owner.tandem_solutions), best_distance))
                new_owner.add_displaced_solution(orphan.solution)
            else:
                if threshold_fn(orphan.solution):
                    if n_chromatograms > 0:
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
                if debug_mode:
                    self.log("... Assigning %s to %s out of %r\n" % (
                        solutions[0], chromatogram, solutions))
                chromatogram.assign_entity(solutions[0], entity_chromatogram_type=entity_chromatogram_type)
                chromatogram.representative_solutions = solutions

    def merge_common_entities(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                              threshold_fn=lambda x: x.q_value < 0.05):
        job = AnnotatedChromatogramAggregator(
            annotated_chromatograms, delta_rt=delta_rt, require_unmodified=require_unmodified,
            threshold_fn=threshold_fn)
        return job.run()

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]
