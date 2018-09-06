import numpy as np

from glycan_profiling.structure import FragmentMatchMap
from glycopeptidepy.structure.fragment import IonSeries

from .base import PeptideSpectrumMatcherBase


class LogIntensityScorer(PeptideSpectrumMatcherBase):
    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(LogIntensityScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.n_theoretical = 0

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
        n_theoretical = 0

        neutral_losses = tuple(kwargs.pop("neutral_losses", []))

        for fragments in self.target.get_fragments(IonSeries.b, neutral_losses):
            for frag in fragments:
                n_theoretical += 1
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)

        for fragments in self.target.get_fragments(IonSeries.y, neutral_losses):
            for frag in fragments:
                n_theoretical += 1
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)

        self.solution_map = solution_map
        self.n_theoretical = n_theoretical

    def _intensity_score(self, error_tolerance=2e-5, *args, **kwargs):
        total = 0
        for peak, fragment in self.solution_map:
            total += np.log10(peak.intensity)
        return total

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        self._score = self._intensity_score(error_tolerance, *args, **kwargs)
        return self._score
