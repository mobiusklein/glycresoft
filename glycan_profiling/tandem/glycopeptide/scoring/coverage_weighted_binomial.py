from ...spectrum_matcher_base import SpectrumMatcherBase

from .binomial_score import BinomialSpectrumMatcher
from .simple_score import SimpleCoverageScorer
from .fragment_match_map import FragmentMatchMap


class CoverageWeightedBinomialScorer(SpectrumMatcherBase):
    def __init__(self, scan, sequence):
        super(CoverageWeightedBinomialScorer, self).__init__(scan, sequence)
        self._sanitized_spectrum = set(self.spectrum)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.n_theoretical = 0
        self._binomial = BinomialSpectrumMatcher(scan, sequence)
        self._coverage = SimpleCoverageScorer(scan, sequence)

    def match(self, *args, **kwargs):
        self._binomial.match(*args, **kwargs)
        self._coverage.match(*args, **kwargs)
        self.n_theoretical = self._binomial.n_theoretical
        self.solution_map = self._binomial.solution_map
        self.glycosylated_b_ion_count = self._coverage.glycosylated_b_ion_count
        self.glycosylated_y_ion_count = self._coverage.glycosylated_y_ion_count

    def calculate_score(self, *args, **kwargs):
        binomial = self._binomial.calculate_score(*args, **kwargs)
        coverage = self._coverage.calculate_score(*args, **kwargs)
        self._score = binomial * coverage
        return self._score

    def annotate(self, *args, **kwargs):
        return self._binomial.annotate(*args, **kwargs)
