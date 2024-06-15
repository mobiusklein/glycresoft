import numpy as np

from glycresoft.structure import FragmentMatchMap

from .base import (
    PeptideSpectrumMatcherBase, ChemicalShift,
    HCDFragmentationStrategy, IonSeries)


class SimpleCountScorer(PeptideSpectrumMatcherBase):
    def __init__(self, scan, sequence, mass_shift=None):
        super(SimpleCountScorer, self).__init__(scan, sequence, mass_shift)
        self._score = None
        self.solution_map = FragmentMatchMap()

    def calculate_score(self, **kwargs):
        self._score = len(self.solution_map)
        return self._score


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
        n_term_ions[0] = 0
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
        return self.compute_coverage()



try:
    _has_c = True
    from glycresoft._c.tandem.tandem_scoring_helpers import _peptide_compute_coverage_vectors, compute_coverage
    SimpleCoverageScorer._compute_coverage_vectors = _peptide_compute_coverage_vectors
    SimpleCoverageScorer.compute_coverage = compute_coverage
except ImportError:
    _has_c = False
