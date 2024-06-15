import array

from typing import (Callable, Collection, DefaultDict,
                    List, TYPE_CHECKING)

import numpy as np

from glycresoft.chromatogram_tree import (
    Unmodified)

from ..spectrum_match import SpectrumMatch, SpectrumSolutionSet

from .base import SolutionEntry, logger, Predicate, always, TargetType, parsimony_sort

if TYPE_CHECKING:
    from .chromatogram import SpectrumMatchSolutionCollectionBase



class RepresenterSelectionStrategy(object):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        pass

    def compute_weights(self, collection: 'SpectrumMatchSolutionCollectionBase', threshold_fn: Callable = always,
                        reject_shifted: bool = False, targets_ignored: Collection = None, require_valid: bool = True) -> List[SolutionEntry]:
        raise NotImplementedError()

    def select(self, representers):
        raise NotImplementedError()

    def sort_solutions(self, representers):
        return parsimony_sort(representers)

    def get_solutions_for_spectrum(self, solution_set: SpectrumSolutionSet, threshold_fn: Callable = always,
                                   reject_shifted: bool = False, targets_ignored: Collection = None, require_valid: bool = True):
        return filter(lambda x: x.valid, solution_set.get_top_solutions(
            d=5,
            reject_shifted=reject_shifted,
            targets_ignored=targets_ignored,
            require_valid=require_valid))

    def __call__(self, collection, threshold_fn: Callable = always, reject_shifted: bool = False,
                 targets_ignored: Collection = None, require_valid: bool = True) -> List[SolutionEntry]:
        return self.compute_weights(
            collection,
            threshold_fn=threshold_fn,
            reject_shifted=reject_shifted,
            targets_ignored=targets_ignored,
            require_valid=require_valid)


class TotalBestRepresenterStrategy(RepresenterSelectionStrategy):
    def compute_weights(self, collection: 'SpectrumMatchSolutionCollectionBase', threshold_fn: Callable = always,
                        reject_shifted: bool = False, targets_ignored: Collection = None, require_valid: bool = True) -> List[SolutionEntry]:
        scores = DefaultDict(float)
        best_scores = DefaultDict(float)
        best_spectrum_match = dict()
        for psm in collection.tandem_solutions:
            if threshold_fn(psm):
                sols = list(self.get_solutions_for_spectrum(
                    psm,
                    reject_shifted=reject_shifted,
                    targets_ignored=targets_ignored,
                    require_valid=require_valid)
                )
                for sol in sols:
                    if not threshold_fn(sol):
                        continue
                    if reject_shifted and sol.mass_shift != Unmodified:
                        continue
                    scores[sol.target] += (sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
                        best_spectrum_match[sol.target] = sol

        if len(scores) == 0 and require_valid and collection.tandem_solutions:
            weights = self.compute_weights(
                collection,
                threshold_fn=threshold_fn,
                reject_shifted=reject_shifted,
                targets_ignored=targets_ignored,
                require_valid=False
            )
            if weights:
                logger.warning(
                    f"Failed to find a valid solution for {collection.tandem_solutions}, falling back to previously disqualified solutions")
            return weights

        total = sum(scores.values())
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k],
                          best_spectrum_match[k]) for k, v in scores.items()
            if k in best_spectrum_match
        ]
        weights = self.sort_solutions(weights)
        return weights


class TotalAboveAverageBestRepresenterStrategy(RepresenterSelectionStrategy):
    def compute_weights(self, collection: 'SpectrumMatchSolutionCollectionBase', threshold_fn: Callable = always,
                        reject_shifted: bool = False, targets_ignored: Collection = None, require_valid: bool = True) -> List[SolutionEntry]:
        scores = DefaultDict(lambda: array.array('d'))
        best_scores = DefaultDict(float)
        best_spectrum_match = dict()
        for psm in collection.tandem_solutions:
            if threshold_fn(psm):
                sols = self.get_solutions_for_spectrum(
                    psm,
                    reject_shifted=reject_shifted,
                    targets_ignored=targets_ignored,
                    require_valid=require_valid
                )
                for sol in sols:
                    if not threshold_fn(sol):
                        continue
                    if reject_shifted and sol.mass_shift != Unmodified:
                        continue
                    scores[sol.target].append(sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
                        best_spectrum_match[sol.target] = sol

        if len(scores) == 0 and require_valid and collection.tandem_solutions:
            weights = self.compute_weights(
                collection,
                threshold_fn=threshold_fn,
                reject_shifted=reject_shifted,
                targets_ignored=targets_ignored,
                require_valid=False
            )
            if weights:
                logger.warning(
                    f"Failed to find a valid solution for {collection.tandem_solutions}, falling back to previously disqualified solutions")
            return weights

        population = np.concatenate(list(scores.values()))
        min_score = np.mean(population) - np.std(population)
        scores = {k: np.array(v) for k, v in scores.items()}
        thresholded_scores = {k: v[v >= min_score].sum()
                              for k, v in scores.items()}

        total = sum([v for v in thresholded_scores.values()])
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k],
                          best_spectrum_match[k]) for k, v in thresholded_scores.items()
            if k in best_spectrum_match
        ]
        weights = self.sort_solutions(weights)
        return weights
