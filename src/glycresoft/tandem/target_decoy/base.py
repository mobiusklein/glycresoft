# -*- coding: utf-8 -*-
import math
import logging

from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union
from collections import defaultdict, namedtuple

import numpy as np
try:
    from matplotlib import pyplot as plt
except (ImportError, RuntimeError):
    plt = None

from glycresoft.task import LoggingMixin
from glycresoft.tandem.spectrum_match import SpectrumSolutionSet, SpectrumMatch

ScoreCell = namedtuple('ScoreCell', ['score', 'value'])

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NearestValueLookUp(object):
    """
    A mapping-like object which simplifies
    finding the value of a pair whose key is nearest
    to a given query.

    .. note::
        Queries exceeding the maximum key will return
        the maximum key's value.
    """

    def __init__(self, items):
        if isinstance(items, dict):
            items = items.items()
        self.items = sorted(
            [ScoreCell(*x) for x in items if not np.isnan(x[0])], key=lambda x: x[0])

    def max_key(self):
        try:
            return self.items[-1][0]
        except IndexError:
            return 0

    def _find_closest_item(self, value, key_index=0):
        array = self.items
        lo = 0
        hi = len(array)
        n = hi

        error_tolerance = 1e-3

        if np.isnan(value):
            return lo

        if lo == hi:
            return lo

        while hi - lo:
            i = (hi + lo) // 2
            x = array[i][key_index]
            err = x - value
            if abs(err) < error_tolerance:
                mid = i
                best_index = mid
                best_error = abs(err)
                i = mid - 1
                while i >= 0:
                    x = array[i][key_index]
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = array[i][key_index]
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif (hi - lo) == 1:
                mid = i
                best_index = mid
                best_error = abs(err)
                i = mid - 1
                while i >= 0:
                    x = array[i][key_index]
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i -= 1
                i = mid + 1
                while i < n:
                    x = array[i][key_index]
                    err = abs(x - value)
                    if err < best_error:
                        best_error = err
                        best_index = i
                    elif err > error_tolerance:
                        break
                    i += 1
                return best_index
            elif x < value:
                lo = i
            elif x > value:
                hi = i

    def get_pair(self, key, key_index=0):
        k = self._find_closest_item(key, key_index) + 1
        return self.items[k]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return "{s.__class__.__name__}({size})".format(
            s=self, size=len(self))

    def __getitem__(self, key):
        return self._get_one(key)

    def _get_sequence(self, key):
        value = [self._get_one(k) for k in key]
        if isinstance(key, np.ndarray):
            value = np.array(value, dtype=float)
        return value

    def _get_one(self, key):
        ix = self._find_closest_item(key)
        if ix >= len(self):
            ix = len(self) - 1
        if ix < 0:
            ix = 0
        pair = self.items[ix]
        return pair[1]


try:
    _NearestValueLookUp = NearestValueLookUp
    from glycresoft._c.tandem.target_decoy import NearestValueLookUp as NearestValueLookUp
except ImportError:
    pass


class ScoreThresholdCounter(object):
    def __init__(self, series, thresholds):
        self.series = self._prepare_series(series)
        self.thresholds = sorted(set(np.round((thresholds), 10)))
        self.counter = defaultdict(int)
        self.counts_above_threshold = None
        self.n_thresholds = len(self.thresholds)
        self.threshold_index = 0
        self.current_threshold = self.thresholds[self.threshold_index]
        self.current_count = 0

        self._i = 0
        self._is_done = False

        self.find_counts()
        self.counts_above_threshold = self.compute_complement()
        self.counter = NearestValueLookUp(self.counter)

    def _prepare_series(self, series):
        return sorted(series, key=lambda x: x.score)

    def advance_threshold(self):
        self.threshold_index += 1
        if self.threshold_index < self.n_thresholds:
            self.current_threshold = self.thresholds[self.threshold_index]
            self.counter[self.current_threshold] = self.current_count
            return True
        else:
            self._is_done = True
            return False

    def test(self, item):
        if item.score < self.current_threshold:
            self.current_count += 1
            self._i += 1
        else:
            # Rather than using recursion, just invert the condition
            # being tested and loop here.
            while self.advance_threshold():
                if item.score > self.current_threshold:
                    continue
                else:
                    self.current_count += 1
                    self._i += 1
                    break

    def find_counts(self):
        for item in self.series:
            self.test(item)

    def compute_complement(self):
        complement = defaultdict(int)
        n = len(self.series)

        for k, v in self.counter.items():
            complement[k] = n - v
        return NearestValueLookUp(complement)


class ArrayScoreThresholdCounter(ScoreThresholdCounter):

    def _prepare_series(self, series):
        return np.sort(np.array(series))

    def test(self, item):
        if item < self.current_threshold:
            self.current_count += 1
            self._i += 1
        else:
            # Rather than using recursion, just invert the condition
            # being tested and loop here.
            while self.advance_threshold():
                if item > self.current_threshold:
                    continue
                else:
                    self.current_count += 1
                    self._i += 1
                    break


class TargetDecoySet(NamedTuple):
    target_matches: List[SpectrumSolutionSet]
    decoy_matches: List[SpectrumSolutionSet]

    def target_count(self):
        return len(self.target_matches)

    def decoy_count(self):
        return len(self.decoy_matches)


# implementation derived from pyteomics
_precalc_fact = np.log([math.factorial(n) for n in range(20)])


def log_factorial(x):
    x = np.array(x)
    m = (x >= _precalc_fact.size)
    out = np.empty(x.shape)
    out[~m] = _precalc_fact[x[~m].astype(int)]
    x = x[m]
    out[m] = x * np.log(x) - x + 0.5 * np.log(2 * np.pi * x)
    return out


def _log_pi_r(d, k, p=0.5):
    return k * math.log(p) + log_factorial(k + d) - log_factorial(k) - log_factorial(d)


def _log_pi(d, k, p=0.5):
    return _log_pi_r(d, k, p) + (d + 1) * math.log(1 - p)


def _expectation(d, t, p=0.5):
    """The conditional tail probability for the negative binomial
    random variable for the number of incorrect target matches

    Parameters
    ----------
    d : int
        The number of decoys retained
    t : int
        The number of targets retained
    p : float, optional
        The parameter :math:`p` of the negative binomial,
        :math:`1 / 1 + (ratio of the target database to the decoy database)`

    Returns
    -------
    float
        The theoretical number of incorrect target matches

    References
    ----------
    Levitsky, L. I., Ivanov, M. V., Lobas, A. A., & Gorshkov, M. V. (2017).
    Unbiased False Discovery Rate Estimation for Shotgun Proteomics Based
    on the Target-Decoy Approach. Journal of Proteome Research, 16(2), 393–397.
    https://doi.org/10.1021/acs.jproteome.6b00144
    """
    if t is None:
        return d + 1
    t = int(t)
    m = np.arange(t + 1, dtype=int)
    pi = np.exp(_log_pi(d, m, p))
    return ((m * pi).cumsum() / pi.cumsum())[t]


def expectation_correction(targets, decoys, ratio):
    """Estimate a correction for the number of decoys at a given
    score threshold for small data size.

    Parameters
    ----------
    targets : int
        The number of targets retained
    decoys : int
        The number of decoys retained
    ratio : float
        The ratio of target database to decoy database

    Returns
    -------
    float
        The number of decoys to add for the correction

    References
    ----------
    Levitsky, L. I., Ivanov, M. V., Lobas, A. A., & Gorshkov, M. V. (2017).
    Unbiased False Discovery Rate Estimation for Shotgun Proteomics Based
    on the Target-Decoy Approach. Journal of Proteome Research, 16(2), 393–397.
    https://doi.org/10.1021/acs.jproteome.6b00144
    """
    p = 1. / (1. + ratio)
    tfalse = _expectation(decoys, targets, p)
    return tfalse


class FDREstimatorBase(LoggingMixin):

    def summarize(self, name: Optional[str] = None):
        if name is None:
            name = "FDR"

        threshold_05, count_05 = self.get_count_for_fdr(0.05)
        self.log(f"5% {name} = {threshold_05:0.3f} ({count_05})")
        threshold_01, count_01 = self.get_count_for_fdr(0.01)
        self.log(f"1% {name} = {threshold_01:0.3f} ({count_01})")

    def score(self, spectrum_match: SpectrumMatch, assign: bool=False) -> float:
        raise NotImplementedError()

    def score_all(self, solution_set: SpectrumSolutionSet):
        raise NotImplementedError()

    def plot(self, ax=None):
        raise NotImplementedError()

    def get_count_for_fdr(self, threshold: float) -> Tuple[float, int]:
        raise NotImplementedError()


class TargetDecoyAnalyzer(FDREstimatorBase):
    """Estimate the False Discovery Rate using the Target-Decoy method.

    Attributes
    ----------
    database_ratio : float
        The ratio of the size of the target database to the decoy database
    target_weight : float
        A weight (less than 1.0) to put on target matches to make them weaker
        than decoys in situations where there is little data.
    decoy_correction : Number
        A quantity to use to correct for decoys, and if non-zero,
        will indicate that the negative binomial correction for decoys should be
        used.
    decoy_pseudocount : Number
        The value to report when querying the decoy count for a score exceeding
        the maximum score of a decoy match. This is distinct from `decoy_correction`
    decoy_count : int
        The total number of decoys
    decoys : list
        The decoy matches to consider
    n_decoys_at : dict
        The number of decoy matches above each threshold
    n_targets_at : dict
        The number of target matches above each threshold
    target_count : int
        The total number of targets
    targets : list
        The target matches to consider
    thresholds : list
        The distinct score thresholds
    with_pit : bool
        Whether or not to use the "percent incorrect target" adjustment
    """
    n_targets_at: NearestValueLookUp
    n_decoys_at: NearestValueLookUp

    targets: Sequence[SpectrumMatch]
    decoys: Sequence[SpectrumMatch]

    target_count: int
    decoy_count: int

    database_ratio: float
    decoy_correction: float

    target_weight: float
    decoy_pseudocount: float

    def __init__(self, target_series, decoy_series, with_pit=False, decoy_correction=0, database_ratio=1.0,
                 target_weight=1.0, decoy_pseudocount=1.0):
        self.targets = target_series
        self.decoys = decoy_series
        self.target_count = len(target_series)
        self.decoy_count = len(decoy_series)
        self.database_ratio = database_ratio
        self.target_weight = target_weight
        self.with_pit = with_pit
        self.decoy_correction = decoy_correction
        self.decoy_pseudocount = decoy_pseudocount

        self._calculate_thresholds()
        self._q_value_map = self.calculate_q_values()

    def get_score(self, spectrum_match: SpectrumMatch) -> float:
        return spectrum_match.score

    def has_score(self, spectrum_match: SpectrumMatch) -> bool:
        return hasattr(spectrum_match, 'score')

    def pack(self):
        self.targets = []
        self.decoys = []

    def _calculate_thresholds(self):
        self.n_targets_at = NearestValueLookUp([])
        self.n_decoys_at = NearestValueLookUp([])

        target_series = self.targets
        decoy_series = self.decoys

        if len(target_series) and self.has_score(target_series[0]):
            target_series = np.array([self.get_score(t)
                                      for t in target_series], dtype=float)
        else:
            target_series = np.array(target_series, dtype=float)

        if len(decoy_series) and self.has_score(decoy_series[0]):
            decoy_series = np.array([self.get_score(t)
                                     for t in decoy_series], dtype=float)
        else:
            decoy_series = np.array(decoy_series, dtype=float)

        thresholds = np.unique(
            np.sort(np.concatenate([target_series, decoy_series]))
        )

        self.thresholds = thresholds
        if len(thresholds) > 0:
            self.n_targets_at = ArrayScoreThresholdCounter(
                target_series, self.thresholds).counts_above_threshold
            self.n_decoys_at = ArrayScoreThresholdCounter(
                decoy_series, self.thresholds).counts_above_threshold
        else:
            self.n_targets_at = NearestValueLookUp([])
            self.n_decoys_at = NearestValueLookUp([])

    def n_decoys_above_threshold(self, threshold: float) -> int:
        try:
            if threshold > self.n_decoys_at.max_key():
                return self.decoy_pseudocount + self.decoy_correction
            return self.n_decoys_at[threshold] + self.decoy_correction
        except (IndexError, KeyError):
            if len(self.n_decoys_at) == 0:
                return self.decoy_correction
            else:
                raise

    def n_targets_above_threshold(self, threshold: float) -> int:
        try:
            return self.n_targets_at[threshold]
        except (IndexError, KeyError):
            if len(self.n_targets_at) == 0:
                return 0
            else:
                raise

    def expectation_correction(self, t: int, d: int) -> float:
        return expectation_correction(t, d, self.database_ratio)

    def target_decoy_ratio(self, cutoff: float) -> float:

        decoys_at = self.n_decoys_above_threshold(cutoff)
        targets_at = self.n_targets_above_threshold(cutoff)
        decoy_correction = 0
        if self.decoy_correction:
            try:
                decoy_correction = self.expectation_correction(
                    targets_at, decoys_at)
            except Exception as ex:
                print(ex)
        try:
            ratio = (decoys_at + decoy_correction) / float(
                targets_at * self.database_ratio * self.target_weight)
        except ZeroDivisionError:
            ratio = (decoys_at + decoy_correction)
        return ratio, targets_at, decoys_at

    def estimate_percent_incorrect_targets(self, cutoff: float) -> float:
        target_cut = self.target_count - self.n_targets_above_threshold(cutoff)
        decoy_cut = self.decoy_count - self.n_decoys_above_threshold(cutoff)
        percent_incorrect_targets = target_cut / float(decoy_cut)

        return percent_incorrect_targets

    def estimate_fdr(self, cutoff: float) -> float:
        if self.with_pit:
            percent_incorrect_targets = self.estimate_percent_incorrect_targets(
                cutoff)
        else:
            percent_incorrect_targets = 1.0
        return percent_incorrect_targets * self.target_decoy_ratio(cutoff)[0]

    def calculate_q_values(self):
        # Thresholds in ascending order
        thresholds = self.thresholds
        mapping = {}
        last_score = float('inf')
        last_q_value = 0
        for threshold in thresholds:
            try:
                q_value = self.estimate_fdr(threshold)
                # If a worse score has a lower q-value than a better score, use that q-value
                # instead.
                if last_q_value < q_value and last_score < threshold:
                    q_value = last_q_value
                last_q_value = q_value
                last_score = threshold
                mapping[threshold] = q_value
            except ZeroDivisionError:
                mapping[threshold] = 1.
        return NearestValueLookUp(mapping)

    def score_for_fdr(self, fdr_estimate: float) -> float:
        i = -1
        n = len(self.q_value_map)
        if n == 0:
            return 0
        for _score, fdr in self.q_value_map.items:
            i += 1
            if fdr_estimate >= fdr:
                if i < n:
                    cella = self.q_value_map.items[i]
                else:
                    cella = ScoreCell(fdr_estimate, 0)

                if i - 1 >= 0:
                    cellb = self.q_value_map.items[i - 1]
                else:
                    cellb = ScoreCell(fdr_estimate, 0)

                if i + 1 < n:
                    cellc = self.q_value_map.items[i + 1]
                else:
                    cellc = ScoreCell(fdr_estimate, 0)
                distance_a = abs(fdr_estimate - cella.value)
                distance_b = abs(fdr_estimate - cellb.value)
                distance_c = abs(fdr_estimate - cellc.value)
                min_distance = min(distance_a, distance_b, distance_c)
                if min_distance == distance_a:
                    return cella.score
                elif min_distance == distance_b:
                    return cellb.score
                else:
                    return cellc.score
        return float('inf')

    def get_count_for_fdr(self, fdr_estimate: float) -> Tuple[float, int]:
        threshold = self.score_for_fdr(fdr_estimate)
        count = self.n_targets_above_threshold(threshold)
        return threshold, count

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        thresholds = sorted(self.thresholds, reverse=False)
        target_counts = np.array(
            [self.n_targets_above_threshold(i) for i in thresholds])
        decoy_counts = np.array([self.n_decoys_above_threshold(i)
                                 for i in thresholds])
        fdr = np.array([self.q_value_map[i] for i in thresholds])
        try:
            at_5_percent = np.where(fdr < 0.05)[0][0]
        except IndexError:
            at_5_percent = -1
        try:
            at_1_percent = np.where(fdr < 0.01)[0][0]
        except IndexError:
            at_1_percent = -1
        line1 = ax.plot(thresholds, target_counts,
                        label='Target', color='steelblue')
        line2 = ax.plot(thresholds, decoy_counts, label='Decoy', color='coral')
        tline5 = ax.vlines(
            thresholds[at_5_percent], 0, np.max(target_counts), linestyle='--', color='green',
            lw=0.75, label='5% FDR')
        tline1 = ax.vlines(
            thresholds[at_1_percent], 0, np.max(target_counts), linestyle='--', color='skyblue',
            lw=0.75, label='1% FDR')
        ax.set_ylabel("# Matches Retained")
        ax.set_xlabel("Score")
        ax2 = ax.twinx()
        line3 = ax2.plot(thresholds, fdr, label='FDR',
                         color='grey', linestyle='--')
        ax2.set_ylabel("FDR")
        ax.legend([line1[0], line2[0], line3[0], tline5, tline1],
                  ['Target', 'Decoy', 'FDR', "5% FDR", "1% FDR"], frameon=False)

        lo, hi = ax.get_ylim()
        lo = max(lo, 0)
        ax.set_ylim(lo, hi)
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(0, hi)

        lo, hi = ax.get_xlim()
        ax.set_xlim(-1, hi)
        lo, hi = ax2.get_xlim()
        ax2.set_xlim(-1, hi)
        return ax

    def q_values(self):
        q_map = self._q_value_map
        if len(q_map) == 0:
            import warnings
            warnings.warn("No FDR estimate what possible.")
            for target in self.targets:
                target.q_value = 0.0
            for decoy in self.decoys:
                decoy.q_value = 0.0
            return
        for target in self.targets:
            try:
                target.q_value = q_map[self.get_score(target)]
            except IndexError:
                target.q_value = 0.0
        for decoy in self.decoys:
            try:
                decoy.q_value = q_map[self.get_score(decoy)]
            except IndexError:
                decoy.q_value = 0.0

    def score(self, spectrum_match, assign=True):
        try:
            q_value = self._q_value_map[self.get_score(spectrum_match)]
        except IndexError:
            import warnings
            warnings.warn("Empty q-value mapping. q-value will be 0.")
            q_value = 0.0
        if assign:
            spectrum_match.q_value = q_value
        return q_value

    def score_all(self, solution_set):
        for spectrum_match in solution_set:
            self.score(spectrum_match, assign=True)
        solution_set.q_value = solution_set.best_solution().q_value
        return solution_set

    @property
    def q_value_map(self):
        return self._q_value_map

    @property
    def fdr_map(self):
        return self._q_value_map


class GroupwiseTargetDecoyAnalyzer(FDREstimatorBase):
    _grouping_labels = None
    targets: Sequence[SpectrumMatch]
    decoys: Sequence[SpectrumMatch]

    groups: List[List[Tuple[List[SpectrumMatch], List[SpectrumMatch]]]]
    group_fits: List[TargetDecoyAnalyzer]
    grouping_functions: List[Callable[[SpectrumMatch], bool]]
    grouping_labels: List[str]

    def __init__(self, target_series, decoy_series, with_pit=False, grouping_functions=None, decoy_correction=0,
                 database_ratio=1.0, target_weight=1.0, decoy_pseudocount=1.0, grouping_labels=None):
        if grouping_functions is None:
            grouping_functions = [lambda x: True]
        if grouping_labels is None:
            grouping_labels = ["Group %d" %
                               i for i in range(1, len(grouping_functions) + 1)]
        self.targets = target_series
        self.decoys = decoy_series
        self.with_pit = with_pit
        self.grouping_labels = grouping_labels
        self.grouping_functions = []
        self.groups = []
        self.group_fits = []
        self.decoy_pseudocount = decoy_pseudocount
        self.decoy_correction = decoy_correction
        self.database_ratio = database_ratio
        self.target_weight = target_weight

        for fn in grouping_functions:
            self.add_group(fn)

        self.partition()

    @property
    def grouping_labels(self):
        if self._grouping_labels is None:
            self._grouping_labels = [
                "Group %d" % i for i in range(1, len(self.grouping_functions) + 1)]
        return self._grouping_labels

    @grouping_labels.setter
    def grouping_labels(self, labels):
        self._grouping_labels = labels

    def pack(self):
        self.targets = []
        self.decoys = []
        self.groups = [[] for g in self.groups]
        for fit in self.group_fits:
            fit.pack()

    def partition(self):
        for target in self.targets:
            i = self.find_group(target)
            self.groups[i][0].append(target)
        for decoy in self.decoys:
            i = self.find_group(decoy)
            self.groups[i][1].append(decoy)
        for group in self.groups:
            fit = TargetDecoyAnalyzer(
                *group, with_pit=self.with_pit,
                decoy_correction=self.decoy_correction,
                database_ratio=self.database_ratio,
                target_weight=self.target_weight,
                decoy_pseudocount=self.decoy_pseudocount)
            self.group_fits.append(fit)

    def add_group(self, fn):
        self.grouping_functions.append(fn)
        self.groups.append(([], []))
        return len(self.groups)

    def find_group(self, spectrum_match):
        for i, fn in enumerate(self.grouping_functions):
            if fn(spectrum_match):
                return i
        return None

    def q_values(self):
        for group in self.group_fits:
            group.q_values()

    def score(self, spectrum_match, assign=True):
        i = self.find_group(spectrum_match)
        fit = self.group_fits[i]
        return fit.score(spectrum_match, assign=assign)

    def score_all(self, solution_set):
        for spectrum_match in solution_set:
            self.score(spectrum_match, assign=True)
        solution_set.q_value = solution_set.best_solution().q_value
        return solution_set

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax2 = ax.twinx()
        lines = []
        labels = []
        for _, (group_fit, label) in enumerate(zip(self.group_fits, self.grouping_labels)):
            thresholds = sorted(group_fit.thresholds, reverse=False)
            target_counts = np.array(
                [group_fit.n_targets_above_threshold(i) for i in thresholds])
            decoy_counts = np.array(
                [group_fit.n_decoys_above_threshold(i) for i in thresholds])

            fdr = np.array([group_fit.q_value_map[i] for i in thresholds])
            line1 = ax.plot(thresholds, target_counts,
                            label='%s Target' % label, )
            lines.append(line1[0])
            labels.append(line1[0].get_label())

            line2 = ax.plot(thresholds, decoy_counts,
                            label='%s Decoy' % label, )
            lines.append(line2[0])
            labels.append(line2[0].get_label())

            line3 = ax2.plot(thresholds, fdr, label='%s FDR' % label,
                             linestyle='--')
            lines.append(line3[0])
            labels.append(line3[0].get_label())

        ax.set_ylabel("# Matches Retained")
        ax.set_xlabel("Score")

        ax2.set_ylabel("FDR")
        ax.legend(lines, labels, frameon=False)

        lo, hi = ax.get_ylim()
        lo = max(lo, 0)
        ax.set_ylim(lo, hi)
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(0, hi)

        lo, hi = ax.get_xlim()
        ax.set_xlim(-1, hi)
        lo, hi = ax2.get_xlim()
        ax2.set_xlim(-1, hi)
        return ax

    def summarize(self, name: Optional[str]=None):
        if name is None:
            name = "FDR"

        for group_name, group_fit in zip(self.grouping_labels, self.group_fits):
            group_fit.summarize(f"{group_name} {name}")


class PeptideScoreTargetDecoyAnalyzer(TargetDecoyAnalyzer):
    """
    A :class:`TargetDecoyAnalyzer` subclass for directly
    handling :class:`~.MultiScoreSpectrumMatch` instances.
    """

    def get_score(self, spectrum_match):
        return spectrum_match.score_set.peptide_score

    def has_score(self, spectrum_match):
        try:
            spectrum_match.score_set.peptide_score
            return True
        except AttributeError:
            return False
