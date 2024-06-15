from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union,
                    TYPE_CHECKING, NamedTuple)

import numpy as np

from ms_deisotope.data_source.scan import ProcessedScan
from ms_deisotope.data_source import ProcessedRandomAccessScanSource

from glycresoft.task import TaskBase

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase
from glycresoft.scoring import ChromatogramSolution

from glycresoft.tandem.target_decoy.base import FDREstimatorBase

from ..spectrum_match import (
    SpectrumMatch,
    SpectrumSolutionSet,
    MultiScoreSpectrumMatch,
    SpectrumMatcherBase
)

from .base import Predicate, TargetType, SolutionEntry

if TYPE_CHECKING:
    from .chromatogram import (
        TandemAnnotatedChromatogram,
        SpectrumMatchSolutionCollectionBase,
        TandemSolutionsWithoutChromatogram
    )
    from glycresoft.scoring.elution_time_grouping.structure import (
        ChromatogramProxy,
        GlycopeptideChromatogramProxy
    )
    from glycresoft.scoring.elution_time_grouping.model import ModelEnsemble as RetentionTimeModelEnsemble


class MS2RevisionValidator(TaskBase):
    threshold_fn: Predicate
    q_value_ratio_threshold: float = 1e9

    def __init__(self, threshold_fn: Predicate):
        self.threshold_fn = threshold_fn

    def has_valid_matches(self, chromatogram: 'TandemAnnotatedChromatogram', member_targets: Set[TargetType]) -> bool:
        has_matches = any([chromatogram.solutions_for(target, threshold_fn=self.threshold_fn)
                           for target in member_targets])
        return has_matches

    def validate_spectrum_match(self, spectrum_match: Union[SpectrumMatch, MultiScoreSpectrumMatch],
                                solution_set: SpectrumSolutionSet) -> bool:
        ## It would be desirable to be able to detect when the difference
        ## in explanations are of such radically different quality that
        ## any consideration of revision is a waste, but this can't be
        ## done reliably as a function of q-value. It might be feasible to do
        ## with score, but such a modification would be difficult to tune for
        ## different scoring algorithms simultaneously. Perhaps if we
        ## had a multidimensional threshold instead of a cascading threshold,
        ## but such a thresholding scheme would need to be estimated
        ## on the fly.
        # best_solution: MultiScoreSpectrumMatch = solution_set.best_solution()
        # if not self.threshold_fn(best_solution):
        #     return True
        # ratio = spectrum_match.q_value / best_solution.q_value
        # pass_threshold = ratio <= self.q_value_ratio_threshold
        # if not pass_threshold:
        #     self.log(
        #         f"... Rejecting revision of {spectrum_match.scan.id} from {best_solution.target} to {spectrum_match.target} due to ID confidence {spectrum_match.q_value}/{best_solution.q_value}")
        # return pass_threshold
        return True

    def can_rewrite_best_matches(self, chromatogram: 'TandemAnnotatedChromatogram', target: TargetType) -> bool:
        # The count of spectra we are able to actually interpret the best spectrum match of
        ssets_to_convert = 0
        # The count of spectra where we can reasonably revise the best match
        ssets_could_convert = 0
        for sset in chromatogram.tandem_solutions:
            if self.threshold_fn(sset.best_solution()):
                ssets_to_convert += 1
            try:
                match = sset.solution_for(target)
                if self.threshold_fn(match) and self.validate_spectrum_match(match, sset):
                    if not match.best_match:
                        ssets_could_convert += 1
            except KeyError:
                continue
        if ssets_to_convert == 0:
            return True
        return (ssets_could_convert / ssets_to_convert) >= 0.5

    def do_rewrite_best_matches(self, chromatogram: 'TandemAnnotatedChromatogram',
                                target: TargetType,
                                invalidated_targets: Set[TargetType]):
        for sset in chromatogram.tandem_solutions:
            try:
                match = sset.solution_for(target)
                if self.threshold_fn(match) and self.validate_spectrum_match(match, sset):
                    if not match.best_match:
                        sset.promote_to_best_match(match)
                else:
                    self.debug(
                        f"... Skipping invalidation of {sset.scan_id!r}, alternative {target} did not pass threshold.")
                    continue
            except KeyError:
                # TODO: Fill in missing match against the preferred target
                self.debug(
                    f"... Skipping invalidation of {sset.scan_id!r}, alternative {target} was not matched.")
                continue

            for invalid_target in invalidated_targets:
                try:
                    match = sset.solution_for(invalid_target)
                except KeyError:
                    continue
                # TODO: Is this really the right way to handle cases with totally
                # different peptide backbones? This should require a minimum of MS2 score/FDR
                # threshold passing
                if match.best_match:
                    self.debug(
                        f"... Revoking best match status of {match.target} for scan {match.scan_id!r}")
                match.best_match = False


class SignalUtilizationMS2RevisionValidator(MS2RevisionValidator):
    utilization_ratio_threshold: float = 0.6

    def __init__(self, threshold_fn: Predicate, utilization_ratio_threshold: float = 0.6):
        super().__init__(threshold_fn)
        self.utilization_ratio_threshold = utilization_ratio_threshold

    def validate_spectrum_match(self, spectrum_match: Union[SpectrumMatch, MultiScoreSpectrumMatch], solution_set: SpectrumSolutionSet) -> bool:
        baseline = super().validate_spectrum_match(spectrum_match, solution_set)
        if not baseline:
            return baseline
        best_solution: MultiScoreSpectrumMatch = solution_set.best_solution()
        if not self.threshold_fn(best_solution):
            return True
        ratio = spectrum_match.score_set.total_signal_utilization / \
            best_solution.score_set.total_signal_utilization
        pass_threshold = ratio >= self.utilization_ratio_threshold
        if baseline and not pass_threshold:
            self.log(
                f"... Rejecting revision of {spectrum_match.scan.id} from {best_solution.target} to {spectrum_match.target} due to signal ratio failing to pass threshold {ratio:0.3f}")
        return pass_threshold

    def has_valid_matches(self, chromatogram: 'TandemAnnotatedChromatogram', targets: Set[TargetType]) -> bool:
        has_passing_case = False
        # Loop over all targets and check if any of them manage to pass the ratio threshold,
        # preventing
        for target in targets:
            for sset in chromatogram.tandem_solutions:
                try:
                    solution: MultiScoreSpectrumMatch = sset.solution_for(
                        target)
                except KeyError:
                    continue

                # Don't bother checking solutions that don't pass FDR threshold predicate
                if not self.threshold_fn(solution):
                    continue

                if self.validate_spectrum_match(solution, sset):
                    # If a match passes the threshold, then something in this solution set is valid
                    has_passing_case = True
                    break
                else:
                    # Otherwise keep checking solution sets
                    continue

            # If we found a valid match, we're done.
            if has_passing_case:
                break

        return has_passing_case


class SpectrumMatchBackFiller(TaskBase):
    scorer: Type[SpectrumMatcherBase]
    match_args: Dict[str, Any]

    spectrum_match_cls: Type[SpectrumMatch]
    fdr_estimator: Optional[FDREstimatorBase]
    threshold_fn: Predicate
    mass_shifts: Optional[List[MassShiftBase]]

    scan_loader: ProcessedRandomAccessScanSource

    def __init__(self, scan_loader: ProcessedRandomAccessScanSource,
                 scorer: Type[SpectrumMatcherBase],
                 spectrum_match_cls: Type[SpectrumMatch],
                 id_maker,
                 threshold_fn: Predicate = lambda x: x.q_value < 0.05,
                 match_args=None,
                 fdr_estimator: Optional[FDREstimatorBase] = None,
                 mass_shifts: Optional[List[MassShiftBase]] = None):
        if match_args is None:
            match_args = {}
        self.scan_loader = scan_loader
        self.scorer = scorer
        self.match_args = match_args
        self.spectrum_match_cls = spectrum_match_cls
        self.id_maker = id_maker
        self.fdr_estimator = fdr_estimator
        self.threshold_fn = threshold_fn
        self.mass_shifts = mass_shifts

    def select_best_mass_shift_for(self, scan: ProcessedScan,
                                   structure: TargetType,
                                   mass_shifts: List[MassShiftBase]) -> Tuple[Optional[MassShiftBase], float]:
        observed_mass = scan.precursor_information.neutral_mass
        theoretical_mass = structure.total_mass
        delta = observed_mass - theoretical_mass

        best_shift = None
        best_shift_error = float('inf')

        for shift in mass_shifts:
            err = delta - shift.mass
            abs_err = abs(err)
            if abs_err < best_shift_error:
                best_shift = shift
                best_shift_error = abs_err
        return best_shift, best_shift_error

    def id_for_structure(self, structure: TargetType, reference: TargetType):
        structure.protein_relation = reference.protein_relation
        structure.id = self.id_maker(structure, reference)
        return structure

    def evaluate_spectrum(self, scan_id: str, structure: TargetType, mass_shifts: List[MassShiftBase]) -> SpectrumMatcherBase:
        scan = self.scan_loader.get_scan_by_id(scan_id)
        best_shift, best_shift_error = self.select_best_mass_shift_for(
            scan, structure, mass_shifts)
        match = self.scorer.evaluate(
            scan, structure, mass_shift=best_shift, **self.match_args)
        return match

    def add_matches_to_solution_set(self, sset: SpectrumSolutionSet,
                                    targets: List[TargetType],
                                    mass_shifts: Optional[List[MassShiftBase]] = None,
                                    should_promote: bool=False) -> List[Tuple[SpectrumSolutionSet, SpectrumMatch, bool]]:
        if mass_shifts is None:
            mass_shifts = self.mass_shifts

        solution_set_match_pairs = []

        scan_id = sset.scan_id
        for inst in targets:
            matched = self.evaluate_spectrum(
                scan_id, inst, mass_shifts)
            match = self.spectrum_match_cls.from_match_solution(
                matched)
            match.scan = sset.scan
            sset.insert(0, match)
            solution_set_match_pairs.append((sset, match, False))

        sset.sort()
        sset.mark_top_solutions()
        if self.fdr_estimator is not None:
            self.fdr_estimator.score_all(sset)

        for inst in targets:
            match = sset.solution_for(inst)
            if self.threshold_fn(match) and should_promote:
                match.best_match = True
                sset.promote_to_best_match(match)
        return solution_set_match_pairs


class SpectrumMatchInvalidatinCallback:
    target_to_find: TargetType
    threshold_fn: Callable[[SpectrumMatch], bool]

    def __init__(self, target_to_find: TargetType, threshold_fn: Callable[[SpectrumMatch], bool]):
        self.target_to_find = target_to_find
        self.threshold_fn = threshold_fn

    def __call__(self, sset: SpectrumSolutionSet, sm: SpectrumMatch) -> bool:
        try:
            preferred_sm = sset.solution_for(self.target_to_find)
            if not self.threshold_fn(preferred_sm):
                return True
            return False
        except KeyError:
            return True


class MatchMarkResult(NamedTuple):
    targets: Set[TargetType]
    match_count: int

    def __add__(self, other: 'MatchMarkResult'):
        if other is None:
            return self
        return self.__class__(
            self.targets | other.targets,
            self.match_count + other.match_count
        )

    @classmethod
    def empty(cls):
        return cls(set(), 0)


InvalidationFilter = Callable[[SpectrumSolutionSet, SpectrumMatch], bool]


class RevisionSummary(NamedTuple):
    accepted: bool
    rejected_matches: Optional[MatchMarkResult] = None
    invalidated_matches: Optional[MatchMarkResult] = None

    @property
    def rejected_match_count(self):
        if not self.rejected_matches:
            return 0
        return self.rejected_matches.match_count

    @property
    def invalidated_match_count(self):
        if not self.invalidated_matches:
            return 0
        return self.invalidated_matches.match_count


class SpectrumMatchUpdater(SpectrumMatchBackFiller):
    """Generate updates to :class:`TandemAnnotatedChromatogram` from chromatogram reviser like an RT predictor.

    This type needs to:
        1. Generate new spectrum matches using the same scorer if a solution set lacks
           a match to the suggested target.
        2. Re-calculate the FDR of the new matches if needed.
        3. Generate a new unique ID for each source peptide that a glycopeptide matching
           the revised glycoform could come from, that is consistent with the previous
           ID space that other glycopeptides used.
        4. Update the assigned entity of each assigned chromatogram so that the correct
           structure is the "best match", discarding the :class:`SolutionEntry` for the
           incorrect assignment *without* deleting the spectrum matches to that themselves.
    """

    spectrum_match_cls: Type[SpectrumMatch]
    threshold_fn: Callable[[SpectrumMatch], bool]
    fdr_estimator: FDREstimatorBase
    match_args: Dict[str, Any]

    retention_time_delta: float
    retention_time_model: 'RetentionTimeModelEnsemble'

    def __init__(self, scan_loader,
                 scorer,
                 spectrum_match_cls,
                 id_maker,
                 threshold_fn=lambda x: x.q_value < 0.05,
                 match_args=None,
                 fdr_estimator=None, retention_time_model=None,
                 retention_time_delta=0.35):
        if match_args is None:
            match_args = {}

        super().__init__(
            scan_loader,
            scorer,
            spectrum_match_cls,
            id_maker,
            threshold_fn,
            match_args,
            fdr_estimator)

        self.retention_time_model = retention_time_model
        self.retention_time_delta = retention_time_delta

    def _find_identical_keys_and_matches(self, chromatogram: 'SpectrumMatchSolutionCollectionBase',
                                         structure: TargetType) -> Tuple[Set[TargetType], List[Tuple[SpectrumSolutionSet, SpectrumMatch]]]:
        structures: Set[TargetType] = set()
        key = str(structure)
        spectrum_matches: List[Tuple[SpectrumSolutionSet, SpectrumMatch]] = []
        for sset in chromatogram.tandem_solutions:
            for sm in sset:
                if str(sm.target) == key:
                    structures.add(sm.target)
                    spectrum_matches.append((sset, sm))
        return structures, spectrum_matches

    def find_identical_peptides(self, chromatogram: 'SpectrumMatchSolutionCollectionBase',
                                structure: TargetType) -> Set[TargetType]:
        structures, _spectrum_matches = self._find_identical_keys_and_matches(
            chromatogram, structure)
        return structures

    def find_identical_peptides_and_mark(self, chromatogram: 'SpectrumMatchSolutionCollectionBase',
                                         structure: TargetType,
                                         best_match: bool = False,
                                         valid: bool = False,
                                         reason: Optional[str] = None,
                                         filter_fn: Optional[InvalidationFilter] = None
                                         ) -> MatchMarkResult:

        structures, spectrum_matches = self._find_identical_keys_and_matches(
            chromatogram, structure)
        k = 0
        for sset, sm in spectrum_matches:
            if filter_fn and filter_fn(sset, sm):
                self.debug(
                    f"... NOT Marking {sm.scan_id} -> {sm.target} Valid = {valid},"
                    f" Best Match = {best_match}; Reason: {reason}")
                continue
            self.debug(
                f"... Marking {sm.scan_id} -> {sm.target} Valid = {valid}, Best Match = {best_match}; Reason: {reason}")
            sm.valid = valid
            sm.best_match = best_match
            # Do not want to invalidate order as that would force re-sorting
            # and that would break any "promoted" solution whose score isn't
            # the top one.
            sset._invalidate(invalidate_order=False)
            k += 1
        return MatchMarkResult(structures, k)

    def make_target_filter(self, target_to_find: TargetType) -> InvalidationFilter:
        filter_fn = SpectrumMatchInvalidatinCallback(
            target_to_find, self.threshold_fn)

        return filter_fn

    def get_spectrum_solution_sets(self, revision: Union['TandemAnnotatedChromatogram',
                                                         'TandemSolutionsWithoutChromatogram',
                                                         'ChromatogramProxy'],
                                   chromatogram: Optional['SpectrumMatchSolutionCollectionBase'] = None,
                                   invalidate_reference: bool = True):
        if chromatogram is None:
            chromatogram = revision.source

        reference = chromatogram.structure
        mass_shifts = revision.mass_shifts
        revised_structure = revision.structure

        if invalidate_reference:
            reference_peptides, _count = self.find_identical_peptides_and_mark(
                chromatogram,
                reference,
                reason=f"superceded by {revision.structure}")
        else:
            reference_peptides = self.find_identical_peptides(
                chromatogram, reference)

        revised_solution_sets = self.collect_matches(
            chromatogram,
            revised_structure,
            mass_shifts,
            reference_peptides,
            should_promote=invalidate_reference)

        if self.fdr_estimator is not None:
            for sset in chromatogram.tandem_solutions:
                self.fdr_estimator.score_all(sset)
        return revised_solution_sets

    def collect_candidate_solutions_for_rt_validation(self, chromatogram: Union['TandemAnnotatedChromatogram',
                                                                                'TandemSolutionsWithoutChromatogram'],
                                                      structure: TargetType,
                                                      rt_score: float,
                                                      overridden: Optional[TargetType] = None) -> Tuple[
                                                        List[SolutionEntry],
                                                        List[SolutionEntry],
                                                        List[SolutionEntry],
                                                        Set[TargetType]
                                                    ]:
        from glycresoft.scoring.elution_time_grouping import GlycopeptideChromatogramProxy
        from .chromatogram import TandemSolutionsWithoutChromatogram

        representers: List[SolutionEntry] = chromatogram.compute_representative_weights(
            self.threshold_fn)

        keep: List[SolutionEntry] = []
        match: List[SolutionEntry] = []
        rejected: List[SolutionEntry] = []
        invalidated: Set[TargetType] = set()
        alternative_rt_scores: Dict[TargetType, float] = {}

        for r in representers:
            if str(r.solution) == overridden:
                rejected.append(r)
            elif str(r.solution) == structure:
                match.append(r)
            else:
                if self.retention_time_model and not isinstance(chromatogram, TandemSolutionsWithoutChromatogram):
                    if r.solution not in alternative_rt_scores:
                        proxy = GlycopeptideChromatogramProxy.from_chromatogram(
                            chromatogram)
                        proxy.structure = r.solution
                        alt_rt_score = alternative_rt_scores[r.solution] = self.retention_time_model.score_interval(
                            proxy, 0.01)
                    else:
                        alt_rt_score = alternative_rt_scores[r.solution]
                    if np.isnan(alt_rt_score) and not np.isnan(rt_score):
                        self.log(
                            f"RT score for alternative {r.solution} is NaN, reference {structure}"
                            f" RT score is {rt_score:0.3f}")
                    elif not np.isnan(alt_rt_score) and np.isnan(rt_score):
                        self.log(
                            f"RT score alternative {r.solution} is {alt_rt_score:0.3f}, reference"
                            f" {structure} RT score is NaN")
                    # If either is NaN, this is always false, so we never invalidate a NaN-predicted case.
                    if (rt_score - alt_rt_score) > self.retention_time_delta:
                        invalidated.add(r.solution)
                        continue
                keep.append(r)

        return match, keep, rejected, invalidated

    def process_revision(self, revision: 'GlycopeptideChromatogramProxy',
                         chromatogram: Union[
                            'TandemAnnotatedChromatogram',
                            'TandemSolutionsWithoutChromatogram'] = None) -> 'TandemAnnotatedChromatogram':

        if chromatogram is None:
            chromatogram = revision.source

        reference = chromatogram.structure
        revised_structure = revision.structure
        assert reference != revised_structure

        _revised_solution_sets = self.get_spectrum_solution_sets(
            revision,
            chromatogram,
            invalidate_reference=True
        )

        representers: List[SolutionEntry] = chromatogram.compute_representative_weights(
            self.threshold_fn)

        revised_rt_score = None
        if self.retention_time_model:
            revised_rt_score = self.retention_time_model.score_interval(
                revision, 0.01)

        match, keep, rejected, invalidated = self.collect_candidate_solutions_for_rt_validation(
            chromatogram, revised_structure, revised_rt_score, reference)

        if not match:
            self.log("... Failed to identify alternative %s for %s passing the threshold" % (
                revised_structure, reference))
            return chromatogram, RevisionSummary(False, None, None)

        alternative_filter_fn = self.make_target_filter(revised_structure)

        rejected_track = MatchMarkResult(set(), 0)
        for reject in rejected:
            marks = self.find_identical_peptides_and_mark(
                chromatogram,
                reject.solution,
                reason=f"Rejecting {reject} in favor of {revised_structure}",
                filter_fn=alternative_filter_fn)
            rejected_track = rejected_track + marks

        invalidated_track = MatchMarkResult(set(), 0)
        for invalid in invalidated:
            marks = self.find_identical_peptides_and_mark(
                chromatogram,
                invalid,
                reason=f"... Invalidating {invalid} in favor of {revised_structure}",
                filter_fn=alternative_filter_fn)
            invalidated_track = invalidated_track + marks

        marks = self.find_identical_peptides_and_mark(
            chromatogram,
            reference,
            reason=f"superceded by {revision.structure}")
        rejected_track = rejected_track + marks

        representers = chromatogram.filter_entries(match + keep)
        # When the input is a ChromatogramSolution, we have to explicitly
        # set the attribute on the wrapped chromatogram object, not the wrapper
        # or else it will be lost during later unwrapping.
        if isinstance(chromatogram, ChromatogramSolution):
            chromatogram.chromatogram.representative_solutions = representers
        else:
            chromatogram.representative_solutions = representers

        # This mutation is safe to call directly since it passes through the
        # universal __getattr__ wrapper
        chromatogram.assign_entity(match[0])
        return chromatogram, RevisionSummary(True, rejected_track, invalidated_track)

    def collect_matches(self,
                        chromatogram: 'TandemAnnotatedChromatogram',
                        structure: TargetType,
                        mass_shifts: List[MassShiftBase],
                        reference_peptides: List[TargetType],
                        should_promote: bool=False) -> List[Tuple[SpectrumSolutionSet, SpectrumMatch, bool]]:
        """Collect spectrum matches to the new `structure` target, and all identical solutions from different sources for all spectra in `chromatogram`.

        Parameters
        ----------
        chromatogram : :class:`~.TandemAnnotatedChromatogram`
            The solution collection being traversed.
        structure : :class:`~.TargetType`
            The new structure we are going to assign to `chromatogram` and wish to
            ensure there are spectrum matches for.
        mass_shifts : :class:`list` of :class:`~.MassShiftBase`
            The mass shifts which are legal to consider for matching `structure` to spectra
            with.
        reference_peptides : :class:`set` of :class:`~.TargetType`
            Other sources of the peptide for `structure` to ensure that all are reported together.
            Relies on :attr:`id_maker` to re-create appropriately keyed glycoforms.
        should_promote : :class:`bool`
            When :const:`True`, matches to ``structure`` are promoted to best-match status if they
            pass the :attr:`threshold_fn`.
        """
        solution_set_match_pairs = []
        if structure.id is None or reference_peptides:
            instances = self.id_maker(structure, reference_peptides)
        else:
            instances = []
        for sset in chromatogram.tandem_solutions:
            try:
                match = sset.solution_for(structure)
                if not match.best_match and self.threshold_fn(match) and should_promote:
                    sset.promote_to_best_match(match)
                solution_set_match_pairs.append((sset, match, True))
            except KeyError:
                solution_set_match_pairs.extend(
                    self.add_matches_to_solution_set(
                        sset,
                        instances,
                        mass_shifts,
                        should_promote=should_promote
                    )
                )
        return solution_set_match_pairs

    def __call__(self, revision, chromatogram=None):
        return self.process_revision(revision, chromatogram=chromatogram)

    def ensure_q_values(self, chromatogram):
        for sset in chromatogram.tandem_solutions:
            for sm in sset:
                if sm.q_value is None and self.fdr_estimator is not None:
                    self.debug("... Computing q-value for %s+%s @ %r" %
                               (sm.target, sm.mass_shift.name, sm.scan_id))
                    self.fdr_estimator.score(sm)

    def affirm_solution(self, chromatogram: 'TandemAnnotatedChromatogram') -> 'TandemAnnotatedChromatogram':
        reference_peptides = self.find_identical_peptides(
            chromatogram, chromatogram.entity)

        _matches = self.collect_matches(
            chromatogram,
            chromatogram.entity,
            chromatogram.mass_shifts,
            reference_peptides,
            should_promote=True
        )
        revised_rt_score = None
        if self.retention_time_model:
            revised_rt_score = self.retention_time_model.score_interval(
                chromatogram, 0.01)

        self.ensure_q_values(chromatogram)
        match, _keep, rejected, invalidated = self.collect_candidate_solutions_for_rt_validation(
            chromatogram, chromatogram.entity, revised_rt_score, None)

        filter_fn = self.make_target_filter(chromatogram.entity)

        rejected_track = MatchMarkResult.empty()
        for reject in rejected:
            marks = self.find_identical_peptides_and_mark(
                chromatogram,
                reject.solution,
                reason=f"Rejecting {reject} in favor of {chromatogram.entity}",
                filter_fn=filter_fn
            )
            rejected_track = rejected_track + marks

        invalidated_track = MatchMarkResult.empty()
        for invalid in invalidated:
            marks = self.find_identical_peptides_and_mark(
                chromatogram,
                invalid,
                reason=f"Invalidating {invalid} in favor of {chromatogram.entity}",
                filter_fn=filter_fn
            )
            invalidated_track = invalidated_track + marks

        if not match:
            self.log("... Failed to affirm %s @ %0.2f passing the threshold" % (
                chromatogram.entity, getattr(chromatogram, "apex_time", -1)))
            affirmed = False
        else:
            affirmed = True
        return chromatogram, RevisionSummary(affirmed, rejected_track, invalidated_track)
