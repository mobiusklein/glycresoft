"""Represent collections of :class:`~SpectrumMatch` instances covering the same spectrum.

Also includes methods for selecting which are worth keeping for downstream consideration.
"""
import logging
from typing import Iterable, Iterator, List, Dict, Any, Optional, Type, Union, Sequence, TypeVar

from ms_deisotope.data_source import ProcessedScan

from glycresoft.chromatogram_tree import Unmodified
from glycresoft.task import log_handle

from .spectrum_match import SpectrumMatch, SpectrumReference, ScanWrapperBase, MultiScoreSpectrumMatch


logger = logging.getLogger(__name__)

MatchType = TypeVar('MatchType', bound=SpectrumMatch)


class SpectrumMatchRetentionStrategyBase(object):
    """
    A method for filtering :class:`SpectrumMatch` objects out of a list according to a specific criterion.

    Attributes
    ----------
    threshold: object
        Some abstract threshold
    """
    threshold: float

    def __init__(self, threshold):
        self.threshold = threshold

    def filter_matches(self, solution_set: 'SpectrumSolutionSet'):
        """
        Filter :class:`SpectrumMatch` objects from a list.

        Parameters
        ----------
        solution_set : list
            The list of :class:`SpectrumMatch` objects to filter.

        Returns
        -------
        list
        """
        raise NotImplementedError()

    def __call__(self, solution_set: 'SpectrumSolutionSet'):
        return self.filter_matches(solution_set)

    def __repr__(self):
        return "{self.__class__.__name__}({self.threshold})".format(self=self)


class MinimumScoreRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    """
    Filtering :class:`~.SpectrumMatch` from a list if their :attr:`~.SpectrumMatch.score` < :attr:`threshold`.

    Parameters
    ----------
    solution_set : list
        The list of :class:`SpectrumMatch` objects to filter.

    Returns
    -------
    list
    """

    def filter_matches(self, solution_set):
        retain = []
        for match in solution_set:
            if match.score > self.threshold:
                retain.append(match)
        return retain


class MinimumMultiScoreRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    """
    A minimum score filtering strategy for multi-score matches.

    A strategy for filtering :class:`~.SpectrumMatch` from a list if
    their :attr:`~.SpectrumMatch.score_set` is less than :attr:`threshold`,
    assuming they share the same dimensions.

    Parameters
    ----------
    solution_set : list
        The list of :class:`SpectrumMatch` objects to filter.

    Returns
    -------
    list
    """

    def filter_matches(self, solution_set):
        retain = []
        for match in solution_set:
            for score_i, ref_i in zip(match.score_set, self.threshold):
                if score_i < ref_i:
                    break
            else:
                retain.append(match)
        return retain


class MaximumSolutionCountRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    """
    A strategy for filtering :class:`~.SpectrumMatch` from a list to retain the top :attr:`threshold` entries.

    This assumes that `solution_set` is sorted.

    Parameters
    ----------
    solution_set : list
        The list of :class:`SpectrumMatch` objects to filter.

    Returns
    -------
    list
    """

    def group_solutions(self, solutions, threshold=1e-2):
        """
        Group solutions which have scores that are very close to one-another so they are not arbitrarily truncated.

        Parameters
        ----------
        solutions : list
            A list of :class:`~.SpectrumMatchBase`
        threshold : float, optional
            The maxmimum distance between two scores to still be considered
            part of a group (the default is 1e-2)

        Returns
        -------
        list
        """
        groups = []
        if len(solutions) == 0:
            return groups
        current_group = [solutions[0]]
        last_solution = solutions[0]
        for solution in solutions[1:]:
            delta = abs(solution.score - last_solution.score)
            if delta > threshold:
                groups.append(current_group)
                current_group = [solution]
            else:
                current_group.append(solution)
            last_solution = solution
        groups.append(current_group)
        return groups

    def filter_matches(self, solution_set):
        groups = self.group_solutions(solution_set)
        return [b for a in groups[:self.threshold] for b in a]


class TopScoringSolutionsRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    """
    Retain only solutions with :attr:`threshold` of the top score.

    A strategy for filtering :class:`~.SpectrumMatch` from a list to retain
    those with scores that are within :attr:`threshold` of the highest score in
    the set.

    This assumes that `solution_set` is sorted and that the highest score is at
    the 0th index..

    Parameters
    ----------
    solution_set : list
        The list of :class:`SpectrumMatch` objects to filter.

    Returns
    -------
    list
    """

    def filter_matches(self, solution_set):
        if len(solution_set) == 0:
            return solution_set
        best_score = solution_set[0].score
        retain = []
        for solution in solution_set:
            if (best_score - solution.score) < self.threshold:
                retain.append(solution)
        return retain


class QValueRetentionStrategy(SpectrumMatchRetentionStrategyBase):
    """
    Filtering :class:`~.SpectrumMatch` from a list if their :attr:`~.SpectrumMatch.q_value` < :attr:`threshold`.

    A strategy for filtering :class:`~.SpectrumMatch` from a list to retain
    those with q-value that are less than :attr:`threshold`.

    Parameters
    ----------
    solution_set : list
        The list of :class:`SpectrumMatch` objects to filter.

    Returns
    -------
    list
    """

    def filter_matches(self, solution_set):
        retain = []
        for match in solution_set:
            if match.q_value < self.threshold:
                retain.append(match)
        return retain


class SpectrumMatchRetentionMethod(SpectrumMatchRetentionStrategyBase):
    """
    Combine several :class:`SpectrumMatchRetentionStrategyBase` together.

    A collection of several :class:`SpectrumMatchRetentionStrategyBase`
    objects which are applied in order to iteratively filter out :class:`SpectrumMatch`
    objects.

    This class implements the same :class:`SpectrumMatchRetentionStrategyBase` API
    so it may be used interchangably with single strategies.

    Attributes
    ----------
    strategies: list
        The list of :class:`SpectrumMatchRetentionStrategyBase`
    """

    def __init__(self, strategies=None):  # pylint: disable=super-init-not-called
        if strategies is None:
            strategies = []
        self.strategies = strategies

    def filter_matches(self, solution_set):
        retained = list(solution_set)
        for strategy in self.strategies:
            retained = strategy(retained)
        return retained

    def __repr__(self):
        return "{self.__class__.__name__}({self.strategies!r})".format(self=self)


default_selection_method = SpectrumMatchRetentionMethod([
    MinimumScoreRetentionStrategy(4.),
    TopScoringSolutionsRetentionStrategy(3.),
    MaximumSolutionCountRetentionStrategy(100)
])


default_multiscore_selection_method = SpectrumMatchRetentionMethod([
    MinimumMultiScoreRetentionStrategy((1.0, 0., 0.)),
    TopScoringSolutionsRetentionStrategy(100.),
    MaximumSolutionCountRetentionStrategy(100),
])

class SpectrumMatchSortingStrategy(object):
    """A base strategy for sorting spectrum matches to rank the best solution."""

    def key_function(self, spectrum_match: SpectrumMatch):
        """
        Create the sorting key for the spectrum match.

        For consistency, the target's ``id`` attribute is used to order
        matches that have the same score so repeated searches with the same
        database produce the same ordering.

        Returns
        -------
        tuple
        """
        return (spectrum_match.valid, spectrum_match.score, spectrum_match.target.id)

    def sort(self, solution_set: 'SpectrumSolutionSet', maximize: bool = True) -> List[SpectrumMatch]:
        """
        Perform the sorting step.

        Parameters
        ----------
        solution_set: :class:`Iterable` of :class:`~.SpectrumMatch` instances
            The spectrum matches to sort
        maximize: bool
            Whether to sort ascending or descending

        Returns
        -------
        :class:`list` of :class:`~.SpectrumMatch` instances
        """
        return sorted(solution_set, key=self.key_function, reverse=maximize)

    def __call__(self, solution_set: 'SpectrumSolutionSet', maximize: bool = True) -> List[SpectrumMatch]:
        """
        Perform the sorting step.

        Parameters
        ----------
        solution_set: :class:`Iterable` of :class:`~.SpectrumMatch` instances
            The spectrum matches to sort
        maximize: bool
            Whether to sort ascending or descending

        Returns
        -------
        :class:`list` of :class:`~.SpectrumMatch` instances
        """
        return self.sort(solution_set, maximize=maximize)


class MultiScoreSpectrumMatchSortingStrategy(SpectrumMatchSortingStrategy):
    def key_function(self, spectrum_match: MultiScoreSpectrumMatch):
        return (spectrum_match.valid, spectrum_match.score_set, spectrum_match.target.id)


class NOParsimonyMixin(object):
    """
    Provides shared methods for a parsimony step re-ordering solutions when the top
    solution may be N-linked or O-linked.
    """

    threshold_percentile: float

    def __init__(self, threshold_percentile=0.5):
        self.threshold_percentile = threshold_percentile

    def get_score(self, solution):
        return solution.score_set.glycan_score

    def get_target(self, solution):
        return solution.target

    def hoist_equivalent_n_linked_solution(self, solution_set: Iterable[SpectrumMatch],
                                           maximize: bool=True) -> List[SpectrumMatch]:
        """
        Apply a parsimony step re-ordering solutions when the top solution may be N-linked or O-linked.

        The O-linked core motif is much simpler than the N-linked core, so it gets
        preferential treatment. This re-orders the matches so that the N-linked solution
        equivalent to the O-linked solution gets marked as the top solution if the
        N-linked solution's glycan score is within :attr:`threshold_percent` of the
        actual top scoring O-linked solution.

        Parameters
        ----------
        solution_set: :class:`Iterable` of :class:`~.SpectrumMatch` instances
            The spectrum matches to sort
        maximize: bool
            Whether to sort ascending or descending

        Returns
        -------
        :class:`list` of :class:`~.SpectrumMatch` instances
        """
        best_solution = solution_set[0]
        best_gp = self.get_target(best_solution)
        best_gc = best_gp.glycan_composition

        hoisted = []
        rest = [best_solution]
        if maximize:
            threshold = self.get_score(best_solution) * self.threshold_percentile
        else:
            threshold = self.get_score(best_solution) * (1 + (1 - self.threshold_percentile))
        i = 0
        for i, sm in enumerate(solution_set):
            if i == 0:
                continue

            if maximize:
                if self.get_score(sm) < threshold:
                    rest.append(sm)
                    break
            else:
                if self.get_score(sm) > threshold:
                    rest.append(sm)
                    break
            sm_target = self.get_target(sm)
            if (best_gp.base_sequence_equality(sm_target) and str(best_gc) == str(sm_target.glycan_composition)
                and sm_target.is_n_glycosylated()):
                hoisted.append(sm)
                log_handle.log(f"Hoisting {sm.target} for scan {sm.scan.scan_id!r}")
            else:
                rest.append(sm)
        rest.extend(solution_set[i + 1:])
        hoisted.extend(rest)
        return hoisted


class NOParsimonyMultiScoreSpectrumMatchSortingStrategy(NOParsimonyMixin, MultiScoreSpectrumMatchSortingStrategy):
    """
    A sorting strategy that applies prefers N-glycans to O-glycans on the same peptide.

    It applies parsimony to selecting the top solution when
    an N-linked solution is more parsimonious than an O-linked solution.
    """

    def sort(self, solution_set: Iterable[MultiScoreSpectrumMatch], maximize=True) -> List[MultiScoreSpectrumMatch]:
        solution_set = super(NOParsimonyMultiScoreSpectrumMatchSortingStrategy, self).sort(solution_set, maximize)
        if solution_set and solution_set[0].target.is_o_glycosylated():
            solution_set = self.hoist_equivalent_n_linked_solution(solution_set, maximize)
        return solution_set


single_score_sorter = SpectrumMatchSortingStrategy()
multi_score_sorter = MultiScoreSpectrumMatchSortingStrategy()
# This sort function isn't useful at this moment. The actual
# ordering is set in chromatogram_mapping
# multi_score_parsimony_sorter = NOParsimonyMultiScoreSpectrumMatchSortingStrategy(0.75)



class SpectrumSolutionSet(ScanWrapperBase, Sequence[MatchType]):
    """
    A collection of spectrum matches against a single scan with different structures.

    Implements the :class:`Sequence` interface.

    Attributes
    ----------
    scan: :class:`~.Scan`-like
        The matched scan
    solutions: list
        The distinct spectrum matches.
    score: float
        The best match's score
    """

    spectrum_match_type: Type[MatchType] = SpectrumMatch
    default_selection_method: SpectrumMatchRetentionMethod = default_selection_method

    scan: Union[ProcessedScan, SpectrumReference]
    solutions: List[MatchType]

    _target_map: Dict[Any, MatchType]
    _is_top_only: bool
    _is_sorted: bool
    _is_simplified: bool
    _q_value: Optional[float]

    def __init__(self, scan, solutions=None):
        if solutions is None:
            solutions = []
        self.scan = scan
        self.solutions = solutions
        self._is_sorted = False
        self._is_simplified = False
        self._is_top_only = False
        self._target_map = None
        self._q_value = None

    def is_ambiguous(self) -> bool:
        seen = set()

        for sm in self:
            if sm.is_best_match:
                seen.add(str(sm.target))
        return len(seen) > 1

    def is_multiscore(self) -> bool:
        """
        Check whether this match has been produced by summarizing a multi-score match, rather
        than a single score match.

        Returns
        -------
        bool
        """
        return False

    def _invalidate(self, invalidate_order: bool):
        if invalidate_order:
            self._is_sorted = False
        self._target_map = None
        self._q_value = None

    @property
    def score(self):
        """The best match's score.

        Returns
        -------
        float
        """
        return self.best_solution().score

    @property
    def q_value(self):
        """The best match's q-value.

        Returns
        -------
        float
        """
        if self._q_value is None:
            self._q_value = self.best_solution().q_value
        return self._q_value

    @q_value.setter
    def q_value(self, value):
        self._q_value = value

    def _make_target_map(self):
        self._target_map = {
            sol.target: sol for sol in self
        }

    def has_solution(self, target) -> bool:
        if self._target_map is None:
            self._make_target_map()
        return target in self._target_map

    def solution_for(self, target) -> MatchType:
        """
        Find the spectrum match from this set which corresponds to the provided `target` structure.

        Parameters
        ----------
        target : object
            The target to search for

        Returns
        -------
        :class:`~.SpectrumMatchBase`
        """
        if self._target_map is None:
            self._make_target_map()
        return self._target_map[target]

    def precursor_mass_accuracy(self):
        """
        The precursor mass accuracy of the best match.

        Returns
        -------
        float
        """
        return self.best_solution().precursor_mass_accuracy()

    def best_solution(self, reject_shifted=False, targets_ignored=None, require_valid=True) -> MatchType:
        """
        The element in :attr:`solutions` which is the best match to :attr:`scan`, the match at position 0.

        If the collection is not sorted, :meth:`sort` will be called.

        Parameters
        ----------
        reject_shifted : bool
            Whether or not to reject any solution where the mass shift is not :obj:`Unmodified`.
            Defaults to :const:`False`.
        targets_ignored : Container
            A collection of :attr:`~.SpectrumMatch.target` values to ignore

        Returns
        -------
        :class:`~.SpectrumMatchBase`
        """
        if not self._is_sorted:
            self.sort()
        if targets_ignored is None:
            targets_ignored = ()
        if not reject_shifted and not targets_ignored:
            for solution in self.solutions:
                if solution.valid:
                    return solution
        for solution in self:
            if (solution.mass_shift == Unmodified or not reject_shifted) and (solution.target in targets_ignored or
                not targets_ignored) and (solution.valid or not require_valid):
                return solution

    def mark_top_solutions(self, reject_shifted=False, targets_ignored=None) -> 'SpectrumSolutionSet':
        solution = self.best_solution(
            reject_shifted=reject_shifted,
            targets_ignored=targets_ignored)
        if solution is None and reject_shifted:
            return self.mark_top_solutions(reject_shifted=False, targets_ignored=targets_ignored)
        if solution is None and targets_ignored:
            return self.mark_top_solutions(
                reject_shifted=reject_shifted, targets_ignored=None)
        if solution is None and not reject_shifted and not targets_ignored:
            logger.warn(f"Could not mark a top solution for {self.scan_id} (not reject shifted and not targets ignore)")
            return self
        if solution is None:
            logger.warn(f"Could not mark a top solution for {self.scan_id}")
            return self
        solution.best_match = True
        best_solution_score = solution.score
        for solution in self:
            if (abs(best_solution_score - solution.score) < 1e-3 or
                best_solution_score < solution.score) and solution.valid:
                solution.best_match = True
            else:
                solution.best_match = False
        return self

    def __repr__(self):
        if len(self) == 0:
            return "{self.__class__.__name__}({self.scan}, [])".format(self=self)
        return "{self.__class__.__name__}({self.scan}, {best.target}, {best.score})".format(
            self=self, best=self.best_solution())

    def __getitem__(self, i):
        return self.solutions[i]

    def __iter__(self) -> Iterator[MatchType]:
        return iter(self.solutions)

    def __len__(self):
        return len(self.solutions)

    def simplify(self):
        """
        Discard excess information in this collection to save space.

        Converts :attr:`scan` to a :class:`~.SpectrumReference`, and
        converts all matches to :class:`~.SpectrumMatch`, discarding
        matcher-specific information.

        """
        if self._is_simplified:
            return
        self.scan = SpectrumReference(
            self.scan.id, self.scan.precursor_information)
        solutions = []
        if len(self) > 0:
            best_score = self.best_solution().score
            for sol in self.solutions:
                sm = self.spectrum_match_type.from_match_solution(sol)
                if abs(sm.score - best_score) < 1e-6:
                    sm.best_match = True
                sm.scan = self.scan
                solutions.append(sm)
            self.solutions = solutions
        self._is_simplified = True
        self._invalidate(False)

    def get_top_solutions(self, d=3, reject_shifted=False, targets_ignored=None,
                          require_valid=True) -> List[MatchType]:
        """
        Get all matches within `d` of the best solution.

        Parameters
        ----------
        d : float, optional
            The delta between the best match and the worst to return
            (the default is 3)
        reject_shifted : bool
            Whether or not to reject any solution where the mass shift is not :obj:`Unmodified`.
            Defaults to :const:`False`.
        targets_ignored : Container
            A collection of :attr:`~.SpectrumMatch.target` values to ignore

        Returns
        -------
        list
        """
        best = self.best_solution(
            reject_shifted=reject_shifted, targets_ignored=targets_ignored, require_valid=require_valid)
        if best is None and reject_shifted:
            return self.get_top_solutions(
                d,
                reject_shifted=False,
                targets_ignored=targets_ignored,
                require_valid=require_valid
            )
        if best is None and targets_ignored:
            return self.get_top_solutions(
                d,
                reject_shifted=reject_shifted,
                targets_ignored=None,
                require_valid=require_valid
            )
        if best is None:
            return []
        best_score = best.score
        if reject_shifted:
            return [x for x in self.solutions if (best_score - x.score) < d and x.mass_shift == Unmodified]
        return [x for x in self.solutions if (best_score - x.score) < d]

    def select_top(self, method=None) -> 'SpectrumSolutionSet':
        """
        Filter spectrum matches in this collection in-place.

        If all the solutions would be filtered out, only the best solution
        will be kept.

        .. warning::
            This method is dangerous to use when spectrum matches in the same
            list are part of different groups (e.g. different decoy sets). Use
            it with caution.

        Parameters
        ----------
        method : :class:`SpectrumRetentionStrategyBase`, optional
            The filtering strategy to use, a callable object that returns a
            shortened list of :class:`~.SpectrumMatchBase` instances.
            If :const:`None`, :attr:`default_selection_method` will be used.

        """
        if method is None:
            method = self.default_selection_method
        if self._is_top_only:
            return self
        if not self._is_sorted:
            self.sort()
        if len(self) > 0:
            best_solution = self.best_solution()
            after = method(self)
            self.solutions = after
            if len(self) == 0:
                self.solutions = [best_solution]
        self._is_top_only = True
        self._invalidate(False)
        return self

    def sort(self, maximize=True, method=None) -> 'SpectrumSolutionSet':
        """
        Sort the spectrum matches in this solution set according to their score attribute.

        In the event of a tie, in order to enforce determistic behavior, this will also
        sort matches according to their target's id attribute.

        Sets :attr:`_is_sorted` to :const:`True`.

        Parameters
        ----------
        maximize : bool, optional
            If true, sort descending order instead of ascending. Defaults to :const:`True`

        See Also
        --------
        sort_by
        sort_q_value
        """
        if method is None:
            method = single_score_sorter
        self.solutions = method(self.solutions, maximize=maximize)
        self._is_sorted = True
        return self

    def sort_by(self, sort_fn=None, maximize=True) -> 'SpectrumSolutionSet':
        """
        Sort the spectrum matches in this solution set according to `sort_fn`.

        This method behaves the same way :meth:`sort` does, except instead of
        sorting on an intrinsic attribute it uses a callable. It uses the same
        determistic augmentation as :meth:`sort`

        Parameters
        ----------
        sort_fn : Callable, optional
            The sort key function to use. If not provided, falls back to :meth:`sort`.
        maximize : bool, optional
            If true, sort descending order instead of ascending. Defaults to :const:`True`

        See Also
        --------
        sort
        """
        if sort_fn is None:
            return self.sort(maximize=maximize)
        self.solutions.sort(key=lambda x: (sort_fn(x), x.target.id), reverse=maximize)
        self._is_sorted = True
        return self

    def sort_q_value(self) -> 'SpectrumSolutionSet':
        """
        Sort the spectrum matches in this solution set according to their q_value attribute.

        In the event of a tie, in order to enforce determistic behavior, this will also
        sort matches according to their target's id attribute.

        Sets :attr:`_is_sorted` to :const:`True`.

        See Also
        --------
        sort
        sort_by
        """
        self.solutions.sort(key=lambda x: (x.q_value, x.target.id), reverse=False)
        self._is_sorted = True
        return self

    def merge(self, other) -> 'SpectrumSolutionSet':
        self._invalidate(True)
        self.solutions.extend(other)
        self.sort()
        if self._is_top_only:
            self._is_top_only = False
            self.select_top()
        return self

    def append(self, match: MatchType) -> 'SpectrumSolutionSet':
        self._invalidate(True)
        self.solutions.append(match)
        self.sort()
        if self._is_top_only:
            self._is_top_only = False
            self.select_top()
        return self

    def insert(self, position: int, match: MatchType, is_sorted=True):
        was_sorted = self._is_sorted
        self._invalidate(True)
        if is_sorted:
            self._is_sorted = was_sorted
        self.solutions.insert(position, match)

    def remove(self, match: MatchType):
        self.solutions.remove(match)

    def promote_to_best_match(self, match: MatchType, reject_shifted: bool=False,
                              targets_ignored: Optional[List[Any]] = None) -> 'SpectrumSolutionSet':
        try:
            self.remove(match)
        except ValueError:
            pass
        self.insert(0, match, is_sorted=True)
        return self.mark_top_solutions(
            reject_shifted=reject_shifted,
            targets_ignored=targets_ignored
        )

    def clone(self):
        dup = self.__class__(self.scan, [
            s.clone() for s in self.solutions
        ])
        dup._is_simplified = self._is_simplified
        dup._is_top_only = self._is_top_only
        dup._is_sorted = self._is_sorted
        return dup

    def __eq__(self, other):
        if self.scan.id != other.scan.id:
            return False
        return self.solutions == other.solutions

    def __ne__(self, other):
        return not (self == other)

    @property
    def key(self) -> frozenset:
        scan_id = self.scan.id
        return frozenset([(scan_id, match.target.id) for match in self])

    def rank(self) -> 'SpectrumSolutionSet':
        """
        Rank the spectrum matches in this solution set by their ordering and scores.

        The best match receiving the lowest rank and successive matches receiving
        higher ranks, the best rank being 1. Spectrum matches with scores that are
        very close together will receive the same rank.

        The score ranking process is only approximate because in some scenarios the
        match ordering is overridden by external factors e.g. retention time, signature
        ions, or adducts.

        Returns
        -------
        SpectrumSolutionSet
        """
        if not self:
            return self
        current_rank = 0
        current_score = float('inf')
        for sm in self:
            if abs(sm.score - current_score) >= 1e-3:
                current_rank += 1
                current_score = sm.score
            sm.rank = current_rank
        return self


def close_to_or_greater_than(value: float, reference: float, delta: float=1e-3):
    return abs(value - reference) < delta or value > reference


class MultiScoreSpectrumSolutionSet(SpectrumSolutionSet[MultiScoreSpectrumMatch]):
    spectrum_match_type = MultiScoreSpectrumMatch
    default_selection_method = default_multiscore_selection_method

    def is_multiscore(self):
        return True

    # NOTE: Sorting by total score is not guaranteed to sort by total
    # FDR. Sorting by FDR after-the-fact is cheating though because we
    # don't count next-best decoys.

    def sort(self, maximize=True, method=None):
        """
        Sort the spectrum matches in this solution set according to their score_set attribute.

        In the event of a tie, in order to enforce determistic behavior, this will also
        sort matches according to their target's id attribute.

        Sets :attr:`_is_sorted` to :const:`True`.

        See Also
        --------
        sort_by
        sort_q_value
        """
        if method is None:
            method = multi_score_sorter
        self.solutions = method(self.solutions, maximize=maximize)
        self._is_sorted = True
        return self

    def mark_top_solutions(self, reject_shifted=False, targets_ignored=None):
        solution = self.best_solution(reject_shifted=reject_shifted, targets_ignored=targets_ignored)
        if solution is None and reject_shifted:
            return self.mark_top_solutions(reject_shifted=False, targets_ignored=targets_ignored)
        if solution is None and targets_ignored:
            return self.mark_top_solutions(reject_shifted=reject_shifted, targets_ignored=None)
        if solution is None and not reject_shifted and not targets_ignored:
            logger.warn(f"Could not mark a top solution for {self.scan_id}")
            return self
        solution.best_match = True
        score_set = solution.score_set
        for solution in self:
            if not solution.valid:
                solution.best_match = False
                continue
            if reject_shifted and solution.mass_shift != Unmodified:
                solution.best_match = False
                continue
            if targets_ignored and solution.target in targets_ignored:
                solution.best_match = False
                continue
            if ((score_set.glycopeptide_score - solution.score_set.glycopeptide_score)) < 1e-3:
                # NOTE: This requires that both the peptide portion and the glycan portion be as
                # good or better than the "best" match. When the "best match" is better in one
                # dimension and worse in another dimension, this becomes more ambiguous. It's
                # possible to query the FDR of the cases, but :meth:`mark_top_solutions` may
                # be called before FDR estimation. Furthermore, a "best match" ranked by FDR
                # potentially introduces all sorts of issues of re-ranking, and could introduce
                # an undesirable number of "alternative" best matches.
                if (score_set.peptide_score - solution.score_set.peptide_score) < 1e-3 and (
                    score_set.glycan_score - solution.score_set.glycan_score) < 1e-3:
                    solution.best_match = True
                else:
                    solution.best_match = False
            else:
                solution.best_match = False
        return self

    def sort_q_value(self):
        """
        Sort the spectrum matches in this solution set according to their q_value_set attribute.

        In the event of a tie, in order to enforce determistic behavior, this will also
        sort matches according to their target's id attribute.

        Sets :attr:`_is_sorted` to :const:`True`.

        See Also
        --------
        sort
        sort_by
        """
        self.solutions.sort(key=lambda x: (
            x.q_value_set, x.score_set, x.target.id), reverse=False)
        self._is_sorted = True
        return self

    @property
    def score_set(self):
        """
        The best match's score set.

        Returns
        -------
        ScoreSet
        """
        return self.best_solution().score_set

    @property
    def q_value_set(self):
        """
        The best match's q-value set.

        Returns
        -------
        FDRSet
        """
        return self.best_solution().q_value_set
