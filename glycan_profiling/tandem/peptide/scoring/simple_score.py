import numpy as np

from glycan_profiling.structure import FragmentMatchMap

from .base import (
    PeptideSpectrumMatcherBase, ChemicalShift,
    HCDFragmentationStrategy, IonSeries)


class SimpleCoverageScorer(PeptideSpectrumMatcherBase):
    def __init__(self, scan, sequence, mass_shift=None):
        super(SimpleCoverageScorer, self).__init__(scan, sequence, mass_shift)
        self._score = None
        self.solution_map = FragmentMatchMap()

    def _compute_coverage_vectors(self):
        n_term_ions = np.zeros(len(self.target))
        c_term_ions = np.zeros(len(self.target))

        for frag in self.solution_map.fragments():
            if frag.series in (IonSeries.b, IonSeries.c):
                n_term_ions[frag.position] = 1
            elif frag.series in (IonSeries.y, IonSeries.z):
                c_term_ions[frag.position] = 1
        return n_term_ions, c_term_ions

    def compute_coverage(self):
        (n_term_ions, c_term_ions) = self._compute_coverage_vectors()

        mean_coverage = np.mean(np.log2(n_term_ions + c_term_ions[::-1] + 1) / np.log2(3))

        return mean_coverage

    def calculate_score(self, **kwargs):
        score = self._coverage_score()
        self._score = score
        return score

    def _coverage_score(self):
        mean_coverage = self.compute_coverage()
        score = mean_coverage
        return score
