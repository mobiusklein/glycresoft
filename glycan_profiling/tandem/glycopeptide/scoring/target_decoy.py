import logging
try:
    logger = logging.getLogger("target_decoy")
except:
    pass
from collections import defaultdict, namedtuple

import numpy as np


ScoreCell = namedtuple('ScoreCell', ['score', 'value'])


def binsearch(array, value):
    lo = 0
    hi = len(array) - 1

    while hi - lo:
        i = (hi + lo) / 2
        x = array[i]
        if x == value:
            return i
        elif hi - lo == 1:
            return i
        elif x < value:
            lo = i
        elif x > value:
            hi = i
    return i


class NearestValueLookUp(object):
    def __init__(self, items):
        if isinstance(items, dict):
            items = items.items()
        self.items = sorted([ScoreCell(*x) for x in items], key=lambda x: x[0])

    def _find_closest_item(self, value):
        array = self.items
        lo = 0
        hi = len(array) - 1

        while hi - lo:
            i = (hi + lo) / 2
            x = array[i][0]
            if x == value:
                return i
            elif (hi - lo) == 1:
                return i
            elif x < value:
                lo = i
            elif x > value:
                hi = i

    def get_pair(self, key):
        return self.items[self._find_closest_item(key) + 1]

    def __getitem__(self, key):
        ix = self._find_closest_item(key)
        ix += 1
        pair = self.items[ix]
        if pair[0] < key:
            return 0
        return pair[1]


class ScoreThresholdCounter(object):
    def __init__(self, series, thresholds):
        self.series = sorted(series, key=lambda x: x.score)
        self.thresholds = sorted(set(np.round((thresholds), 10)))
        self.counter = defaultdict(int)
        self.counts_above_threshold = None
        self.n_thresholds = len(self.thresholds)
        self.threshold_index = 0
        self.current_threshold = thresholds[self.threshold_index]
        self.current_count = 0

        self._i = 0
        self._is_done = False

        self.find_counts()
        self.counts_above_threshold = self.compute_complement()
        self.counter = NearestValueLookUp(self.counter)

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


class TargetDecoyAnalyzer(object):
    def __init__(self, target_series, decoy_series, with_pit=False):
        self.targets = target_series
        self.decoys = decoy_series
        self.target_count = len(target_series)
        self.decoy_count = len(decoy_series)
        self.with_pit = with_pit
        self.calculate_thresholds()
        self._q_value_map = self._calculate_q_values()

    def calculate_thresholds(self):
        self.n_targets_at = {}
        self.n_decoys_at = {}

        target_series = self.targets
        decoy_series = self.decoys

        thresholds = sorted({case.score for case in target_series} | {case.score for case in decoy_series})
        self.thresholds = thresholds

        self.n_targets_at = ScoreThresholdCounter(target_series, thresholds).counts_above_threshold
        self.n_decoys_at = ScoreThresholdCounter(decoy_series, thresholds).counts_above_threshold

    def n_decoys_above_threshold(self, threshold):
        return self.n_decoys_at[threshold]

    def n_targets_above_threshold(self, threshold):
        return self.n_targets_at[threshold]

    def target_decoy_ratio(self, cutoff):

        decoys_at = self.n_decoys_above_threshold(cutoff)
        targets_at = self.n_targets_above_threshold(cutoff)
        try:
            ratio = decoys_at / float(targets_at)
        except ZeroDivisionError:
            ratio = decoys_at
        return ratio, targets_at, decoys_at

    def estimate_percent_incorrect_targets(self, cutoff):
        target_cut = self.target_count - self.n_targets_above_threshold(cutoff)
        decoy_cut = self.decoy_count - self.n_decoys_above_threshold(cutoff)
        percent_incorrect_targets = target_cut / float(decoy_cut)

        return percent_incorrect_targets

    def fdr_with_percent_incorrect_targets(self, cutoff):
        if self.with_pit:
            percent_incorrect_targets = self.estimate_percent_incorrect_targets(cutoff)
        else:
            percent_incorrect_targets = 1.0
        return percent_incorrect_targets * self.target_decoy_ratio(cutoff)[0]

    def _calculate_q_values(self):
        thresholds = sorted(self.thresholds, reverse=False)
        mapping = {}
        last_score = float('inf')
        last_q_value = 0
        for threshold in thresholds:
            try:
                q_value = self.fdr_with_percent_incorrect_targets(threshold)
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

    def q_values(self):
        q_map = self._q_value_map
        for target in self.targets:
            target.q_value = q_map[target.score]
        for decoy in self.decoys:
            decoy.q_value = q_map[decoy.score]

    def score(self, spectrum_match):
        spectrum_match.q_value = self._q_value_map[spectrum_match.score]
        return spectrum_match
