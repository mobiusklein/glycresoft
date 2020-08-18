import numpy as np

from glycan_profiling.structure import FragmentMatchMap
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
