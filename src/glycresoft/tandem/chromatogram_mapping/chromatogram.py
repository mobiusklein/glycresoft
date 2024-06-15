
from typing import (Any, Callable, Collection, DefaultDict,
                    List, Optional,
                    TYPE_CHECKING)

from glycresoft.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycresoft.chromatogram_tree.mass_shift import MassShiftBase
from glycresoft.chromatogram_tree import (
    ChromatogramWrapper,
    Unmodified)

from glycresoft.task import log_handle

from ..spectrum_match import SpectrumMatch, SpectrumSolutionSet

from .base import SolutionEntry, Predicate, always, TargetType
from .representer import RepresenterSelectionStrategy, TotalBestRepresenterStrategy


class SpectrumMatchSolutionCollectionBase(object):
    tandem_solutions: List[SpectrumSolutionSet]
    best_msms_score: float
    representative_solutions: List[SolutionEntry]

    def compute_representative_weights(self, threshold_fn: Predicate = always, reject_shifted: bool = False,
                                       targets_ignored: Collection = None,
                                       strategy: Optional[RepresenterSelectionStrategy] = None, **kwargs) -> List[SolutionEntry]:
        """Calculate a total score for all matched structures across all time points for this
        solution collection, and rank them.

        This total score is the sum of the score over all spectrum matches for which that
        structure was the best match. The percentile is the ratio of the total score for the
        `i`th structure divided by the sum of total scores over all structures.

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
        if strategy is None:
            strategy = TotalBestRepresenterStrategy()
        weights = strategy(self, threshold_fn=threshold_fn,
                           reject_shifted=reject_shifted, targets_ignored=targets_ignored, **kwargs)
        return weights

    def filter_entries(self, entries: List[SolutionEntry], percentile_threshold: float = 1e-5) -> List[SolutionEntry]:
        # This difference is not using the absolute value to allow for scenarios where
        # a worse percentile is located at position 0 e.g. when hoisting via parsimony.
        representers = [x for x in entries if (
            entries[0].percentile - x.percentile) < percentile_threshold]
        return representers

    def most_representative_solutions(self, threshold_fn: Predicate = always, reject_shifted: bool = False,
                                      targets_ignored: Collection = None,
                                      percentile_threshold: float = 1e-5) -> List[SolutionEntry]:
        """Find the most representative solutions, the (very nearly the same, hopefully) structures with
        the highest aggregated score across all MSn events assigned to this collection.

        Parameters
        ----------
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False
        percentile_threshold : float, optional
            The difference between the worst and best percentile to be reported. Defaults to 1e-5.

        Returns
        -------
        list
            A list of solutions with approximately the greatest weight
        """
        weights = self.compute_representative_weights(
            threshold_fn, reject_shifted=reject_shifted, targets_ignored=targets_ignored)
        if weights:
            representers = self.filter_entries(
                weights, percentile_threshold=percentile_threshold)
            return representers
        else:
            return []

    def solutions_for(self, structure: TargetType, threshold_fn: Predicate = always,
                      reject_shifted: bool = False) -> List[SpectrumMatch]:
        '''Get all spectrum matches in this collection for a given
        structure.

        Parameters
        ----------
        structure : Hashable
            The structure collect matches for.
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False

        Returns
        -------
        list
        '''
        solutions = []
        for sset in self.tandem_solutions:
            try:
                psm = sset.solution_for(structure)
                if threshold_fn(psm):
                    if psm.mass_shift != Unmodified and reject_shifted:
                        continue
                    solutions.append(psm)
            except KeyError:
                continue
        return solutions

    def best_match_for(self, structure: TargetType, threshold_fn: Predicate = always) -> SpectrumMatch:
        solutions = self.solutions_for(structure, threshold_fn=threshold_fn)
        if not solutions:
            raise KeyError(structure)
        best_score = -float('inf')
        best_match = None
        for sol in solutions:
            if best_score < sol.score:
                best_score = sol.score
                best_match = sol
        return best_match

    def has_scan(self, scan_id: str) -> bool:
        return any([sset.scan_id == scan_id for sset in self.tandem_solutions])

    def get_scan(self, scan_id: str) -> SpectrumMatch:
        for sset in self.tandem_solutions:
            if sset.scan_id == scan_id:
                return sset
        raise KeyError(scan_id)


class TandemAnnotatedChromatogram(ChromatogramWrapper, SpectrumMatchSolutionCollectionBase):
    time_displaced_assignments: List[SpectrumSolutionSet]

    def __init__(self, chromatogram):
        super(TandemAnnotatedChromatogram, self).__init__(chromatogram)
        self.tandem_solutions = []
        self.time_displaced_assignments = []
        self.best_msms_score = None
        self.representative_solutions = None

    def bisect_charge(self, charge):
        new_charge, new_no_charge = map(
            self.__class__, self.chromatogram.bisect_charge(charge))
        for hit in self.tandem_solutions:
            if hit.precursor_information.charge == charge:
                new_charge.add_solution(hit)
            else:
                new_no_charge.add_solution(hit)
        return new_charge, new_no_charge

    def bisect_mass_shift(self, mass_shift):
        new_mass_shift, new_no_mass_shift = map(
            self.__class__, self.chromatogram.bisect_mass_shift(mass_shift))
        for hit in self.tandem_solutions:
            if hit.best_solution().mass_shift == mass_shift:
                new_mass_shift.add_solution(hit)
            else:
                new_no_mass_shift.add_solution(hit)
        return new_mass_shift, new_no_mass_shift

    def split_sparse(self, delta_rt=1.0):
        parts = list(
            map(self.__class__, self.chromatogram.split_sparse(delta_rt)))
        for hit in self.tandem_solutions:
            nearest = None
            nearest_time = None
            time = hit.scan_time
            for part in parts:
                time_err = min(abs(part.start_time - time),
                               abs(part.end_time - time))
                if time_err < nearest_time:
                    nearest = part
                    nearest_time = time_err
                if part.spans_time_point(time):
                    part.add_solution(hit)
                    break
            else:
                nearest.add_solution(hit)
        return parts

    def add_solution(self, item: SpectrumSolutionSet):
        case_mass = item.precursor_information.neutral_mass
        if abs(case_mass - self.chromatogram.neutral_mass) > 100:
            log_handle.log(
                "Warning, mis-assigned spectrum match to chromatogram %r, %r" % (self, item))
        self.tandem_solutions.append(item)

    def add_displaced_solution(self, item: SpectrumSolutionSet):
        self.add_solution(item)

    def clone(self) -> 'TandemAnnotatedChromatogram':
        new = super(TandemAnnotatedChromatogram, self).clone()
        new.tandem_solutions = list(self.tandem_solutions)
        new.time_displaced_assignments = list(self.time_displaced_assignments)
        new.best_msms_score = self.best_msms_score
        return new

    def merge(self, other: 'TandemAnnotatedChromatogram', node_type: MassShiftBase = Unmodified) -> 'TandemAnnotatedChromatogram':
        if isinstance(other, TandemAnnotatedChromatogram) or hasattr(other, 'tandem_solutions'):
            new = self.__class__(self.chromatogram.merge(
                other.chromatogram, node_type=node_type))
            new.tandem_solutions = self.tandem_solutions + other.tandem_solutions
            new.time_displaced_assignments = self.time_displaced_assignments + \
                other.time_displaced_assignments
        else:
            new = self.chromatogram.merge(other, node_type=node_type)
        return new

    def merge_in_place(self, other: 'TandemAnnotatedChromatogram', node_type: MassShiftBase = Unmodified):
        if isinstance(other, TandemAnnotatedChromatogram) or hasattr(other, 'tandem_solutions'):
            new = self.chromatogram.merge(
                other.chromatogram, node_type=node_type)
            self.tandem_solutions = self.tandem_solutions + other.tandem_solutions
            self.time_displaced_assignments = self.time_displaced_assignments + \
                other.time_displaced_assignments
        else:
            new = self.chromatogram.merge(other, node_type=node_type)
        self.chromatogram = new

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
    # For mimicking EntityChromatograms
    entity: Any
    composition: Any

    mass_shift: MassShiftBase

    def __init__(self, entity, tandem_solutions):
        self.entity = entity
        self.composition = entity
        self.tandem_solutions = tandem_solutions
        self.mass_shift = None
        self.best_msms_score = None
        self.update_mass_shift_and_score()

    def update_mass_shift_and_score(self):
        match = self.best_match_for(self.structure)
        if match is not None:
            self.mass_shift = match.mass_shift
            self.best_msms_score = match.score

    def assign_entity(self, solution_entry: SolutionEntry):
        entity = solution_entry.solution
        mass_shift = solution_entry.match.mass_shift
        self.structure = entity
        self.mass_shift = mass_shift
        self.best_msms_score = solution_entry.best_score

    @property
    def mass_shifts(self):
        return [self.mass_shift]

    @property
    def structure(self):
        return self.entity

    @structure.setter
    def structure(self, value):
        self.entity = value
        self.composition = value

    def __repr__(self):
        template = "{self.__class__.__name__}({self.entity!r}, {self.tandem_solutions}, {self.mass_shift})"
        return template.format(self=self)

    @classmethod
    def aggregate(cls, solutions: List[SpectrumSolutionSet]) -> List['TandemSolutionsWithoutChromatogram']:
        collect = DefaultDict(list)
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
    solution: SpectrumSolutionSet
    scan_time: float

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
