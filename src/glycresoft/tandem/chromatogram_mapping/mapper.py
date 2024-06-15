from typing import (Callable, Collection,
                    List, Optional, Type, Union)

from ms_deisotope.peak_dependency_network.intervals import IntervalTreeNode

from glycresoft.chromatogram_tree import (
    ChromatogramFilter, Chromatogram, build_rt_interval_tree, GlycopeptideChromatogram)

from glycresoft.task import TaskBase

from ..spectrum_match import SpectrumSolutionSet

from .base import default_threshold, Predicate, DEBUG_MODE
from .revision import MS2RevisionValidator
from .chromatogram import TandemAnnotatedChromatogram, TandemSolutionsWithoutChromatogram, ScanTimeBundle
from .aggregation import GraphAnnotatedChromatogramAggregator


class ChromatogramMSMSMapper(TaskBase):
    chromatograms: List[TandemAnnotatedChromatogram]
    orphans: Union[List[ScanTimeBundle],
                   List[TandemSolutionsWithoutChromatogram]]
    error_tolerance: float
    rt_tree: IntervalTreeNode
    scan_id_to_rt: Callable[[str], float]

    def __init__(self, chromatograms: Collection[Chromatogram], error_tolerance: float=1e-5, scan_id_to_rt=lambda x: x):
        self.chromatograms = ChromatogramFilter(map(
            TandemAnnotatedChromatogram, chromatograms))
        self.rt_tree = build_rt_interval_tree(self.chromatograms)
        self.scan_id_to_rt = scan_id_to_rt
        self.orphans = []
        self.error_tolerance = error_tolerance

    def find_chromatogram_spanning(self, time: float):
        return ChromatogramFilter([interv[0] for interv in self.rt_tree.contains_point(time)])

    def find_chromatogram_for(self, solution: SpectrumSolutionSet):
        try:
            precursor_scan_time = self.scan_id_to_rt(
                solution.precursor_information.precursor_scan_id)
        except Exception:
            precursor_scan_time = self.scan_id_to_rt(solution.scan_id)
        overlapping_chroma = self.find_chromatogram_spanning(
            precursor_scan_time)
        chromas = overlapping_chroma.find_all_by_mass(
            solution.precursor_information.neutral_mass, self.error_tolerance)
        if len(chromas) == 0:
            self.orphans.append(ScanTimeBundle(solution, precursor_scan_time))
        else:
            if len(chromas) > 1:
                chroma = max(chromas, key=lambda x: x.total_signal)
            else:
                chroma = chromas[0]
            chroma.tandem_solutions.append(solution)

    def assign_solutions_to_chromatograms(self, solutions: Collection[SpectrumSolutionSet]):
        n = len(solutions)
        for i, solution in enumerate(solutions):
            if i % 5000 == 0:
                self.log("... %d/%d Solutions Handled (%0.2f%%)" %
                         (i, n, (i * 100.0 / n)))
            self.find_chromatogram_for(solution)
        if DEBUG_MODE:
            breakpoint()

    def distribute_orphans(self, threshold_fn: Predicate=default_threshold):
        lost = []
        n = len(self.orphans)
        n_chromatograms = len(self.chromatograms)
        for j, orphan in enumerate(self.orphans):
            mass = orphan.solution.precursor_ion_mass
            time = orphan.scan_time

            if j % 5000 == 0:
                self.log("... %r %d/%d Orphans Handled (%0.2f%%)" %
                         (orphan, j, n, (j * 100.0 / n)))

            candidates = self.chromatograms.find_all_by_mass(
                mass, self.error_tolerance)
            if len(candidates) > 0:
                best_index = 0
                best_distance = float('inf')
                for i, candidate in enumerate(candidates):
                    dist = min(abs(candidate.start_time - time),
                               abs(candidate.end_time - time))
                    if dist < best_distance:
                        best_index = i
                        best_distance = dist
                new_owner = candidates[best_index]

                if best_distance > 5:
                    if threshold_fn(orphan.solution):
                        lost.append(orphan.solution)
                    continue

                if DEBUG_MODE:
                    self.log("... Assigning %r to %r with %d existing solutions with distance %0.3f" %
                             (orphan, new_owner, len(new_owner.tandem_solutions), best_distance))
                new_owner.add_displaced_solution(orphan.solution)
            else:
                if threshold_fn(orphan.solution):
                    if n_chromatograms > 0 and DEBUG_MODE:
                        self.log("No chromatogram found for %r, q-value %0.4f (mass: %0.4f, time: %0.4f)" % (
                            orphan, orphan.solution.q_value, mass, time))
                    lost.append(orphan.solution)
        self.log("Distributed %d orphan identifications, %d did not find a nearby chromatogram" % (
            n, len(lost)))
        self.orphans = TandemSolutionsWithoutChromatogram.aggregate(lost)

    def assign_entities(self, threshold_fn: Predicate=default_threshold,
                        entity_chromatogram_type: Type[Chromatogram]=None):
        if entity_chromatogram_type is None:
            entity_chromatogram_type = GlycopeptideChromatogram
        for chromatogram in self:
            solutions = chromatogram.most_representative_solutions(
                threshold_fn)
            if solutions:
                solutions = sorted(
                    solutions, key=lambda x: x.score, reverse=True)
                chromatogram.assign_entity(
                    solutions[0], entity_chromatogram_type=entity_chromatogram_type)
                chromatogram.representative_solutions = solutions
        if DEBUG_MODE:
            breakpoint()

    def merge_common_entities(self, annotated_chromatograms: List[TandemAnnotatedChromatogram],
                              delta_rt: float = 0.25,
                              require_unmodified: bool = True,
                              threshold_fn: Predicate = default_threshold,
                              revision_validator: Optional[MS2RevisionValidator] = None):
        job = GraphAnnotatedChromatogramAggregator(
            annotated_chromatograms,
            delta_rt=delta_rt,
            require_unmodified=require_unmodified,
            threshold_fn=threshold_fn,
            revision_validator=revision_validator
        )
        result = job.run()
        return result

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]
