import numpy as np
import math

from glycresoft.structure import FragmentMatchMap
from glycopeptidepy.structure.fragment import IonSeries

from .base import PeptideSpectrumMatcherBase
from .simple_score import SimpleCoverageScorer


class LogIntensityScorer(SimpleCoverageScorer):
    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(LogIntensityScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)

    def _intensity_score(self, error_tolerance=2e-5, *args, **kwargs):
        total = 0
        target_ion_series = {
            IonSeries.b, IonSeries.y, IonSeries.c, IonSeries.z}
        for peak, fragment in self.solution_map:
            if fragment.series not in target_ion_series:
                continue
            total += np.log10(peak.intensity)
        return total

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        self._score = self._intensity_score(error_tolerance, *args, **kwargs) * self._coverage_score()
        return self._score


class HyperscoreScorer(PeptideSpectrumMatcherBase):
    def __init__(self, scan, sequence, mass_shift=None):
        super(HyperscoreScorer, self).__init__(scan, sequence, mass_shift)
        self._score = None
        self.solution_map = FragmentMatchMap()

    def _calculate_hyperscore(self, *args, **kwargs):
        n_term_intensity = 0
        c_term_intensity = 0
        n_term = 0
        c_term = 0
        for peak, fragment in self.solution_map:
            if fragment.series in (IonSeries.b, IonSeries.c):
                n_term += 1
                n_term_intensity += peak.intensity
            elif fragment.series in (IonSeries.y, IonSeries.z):
                c_term += 1
                c_term_intensity += peak.intensity
        hyper = 0
        factors = [math.factorial(n_term), math.factorial(c_term),
                   n_term_intensity, c_term_intensity]
        for f in factors:
            if not f:
                continue
            hyper += math.log(f)

        return hyper

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        self._score = self._calculate_hyperscore(
            error_tolerance, *args, **kwargs)
        return self._score


try:
    _has_c = True
    from glycresoft._c.tandem.tandem_scoring_helpers import _calculate_hyperscore
    HyperscoreScorer._calculate_hyperscore = _calculate_hyperscore
except ImportError:
    _has_c = False
