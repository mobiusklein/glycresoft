import numpy as np
import math

from glycopeptidepy.structure.fragment import IonSeries

from .base import ModelTreeNode
from .precursor_mass_accuracy import MassAccuracyModel
from .simple_score import SignatureAwareCoverageScorer


class LogIntensityScorer(SignatureAwareCoverageScorer):
    accuracy_bias = MassAccuracyModel(-2.673807e-07, 5.022458e-06)

    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(LogIntensityScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)

    def _intensity_score(self, error_tolerance=2e-5, *args, **kwargs):
        total = 0
        seen = set()
        for peak, fragment in self.solution_map:
            if peak.index.neutral_mass in seen:
                continue
            seen.add(peak.index.neutral_mass)
            total += np.log10(peak.intensity)
        return total

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        coverage_score = self._coverage_score(*args, **kwargs)
        intensity_score = self._intensity_score(error_tolerance, *args, **kwargs)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = self._signature_ion_score(error_tolerance)
        self._score = intensity_score * coverage_score + mass_accuracy + signature_component

        return self._score

    def peptide_score(self, *args, **kwargs):
        intensity = 0
        seen = set()
        series_set = (IonSeries.b, IonSeries.y, IonSeries.c, IonSeries.z)
        for peak, fragment in self.solution_map:
            if fragment.series in series_set and peak.index.neutral_mass not in seen:
                seen.add(peak.index.neutral_mass)
                intensity += np.log10(peak.intensity)
        n_term, c_term = self._compute_coverage_vectors()[:2]
        coverage_score = ((n_term + c_term[::-1])).sum() / float(self.n_theoretical)
        return intensity * coverage_score

    def glycan_score(self, error_tolerance=2e-5, *args, **kwargs):
        theoretical_set = list(self.target.stub_fragments(extended=True))
        core_fragments = set()
        for frag in theoretical_set:
            if not frag.is_extended:
                core_fragments.add(frag.name)
        core_matches = []
        extended_matches = []
        intensity = 0
        for peak, fragment in self.solution_map:
            if fragment.series == 'stub_glycopeptide':
                if fragment.name in core_fragments:
                    core_matches.append(1.0)
                else:
                    extended_matches.append(1.0)
            intensity += np.log10(peak.intensity)
        core_coverage = (sum(core_matches) ** 2) / len(core_fragments)
        extended_coverage = (
            sum(extended_matches) + sum(core_matches)) / (
                sum(self.target.glycan_composition.values()))
        signature = self._signature_ion_score(error_tolerance)
        # Unlike peptide coverage, the glycan composition coverage operates as a bias towards
        # selecting matches which contain more reliable glycan Y ions, but not to act as a scaling
        # factor because the set of all possible fragments for the glycan composition is a much larger
        # superset of the possible fragments of glycan structures because of recurring patterns
        # not reflected in the glycan composition.
        coverage = core_coverage * extended_coverage
        return intensity * coverage + signature


class ShortPeptideLogIntensityScorer(LogIntensityScorer):
    stub_weight = 0.65


def _short_peptide_test(scan, target, *args, **kwargs):
    return len(target) < 10


LogIntensityModelTree = ModelTreeNode(LogIntensityScorer, {
    _short_peptide_test: ModelTreeNode(ShortPeptideLogIntensityScorer, {}),
})


class HyperscoreScorer(SignatureAwareCoverageScorer):
    accuracy_bias = MassAccuracyModel(-2.673807e-07, 5.022458e-06)

    def _calculate_hyperscore(self, *args, **kwargs):
        n_term_intensity = 0
        c_term_intensity = 0
        stub_intensity = 0
        n_term = 0
        c_term = 0
        stub_count = 0
        for peak, fragment in self.solution_map:
            if fragment.series == "oxonium_ion":
                continue
            elif fragment.series == IonSeries.stub_glycopeptide:
                stub_count += 1
                stub_intensity += peak.intensity
            elif fragment.series in (IonSeries.b, IonSeries.c):
                n_term += 1
                n_term_intensity += peak.intensity
            elif fragment.series in (IonSeries.y, IonSeries.z):
                c_term += 1
                c_term_intensity += peak.intensity
        hyper = 0
        factors = [math.factorial(n_term), math.factorial(c_term), math.factorial(stub_count),
                   stub_intensity, n_term_intensity, c_term_intensity]
        for f in factors:
            hyper += math.log(f)

        return hyper

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        hyperscore = self._calculate_hyperscore(error_tolerance, *args, **kwargs)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = self._signature_ion_score(error_tolerance)
        self._score = hyperscore + mass_accuracy + signature_component

        return self._score
