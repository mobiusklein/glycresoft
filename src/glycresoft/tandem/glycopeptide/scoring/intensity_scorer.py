import math
import numpy as np

from glycopeptidepy.structure.fragment import IonSeries

from .base import ModelTreeNode
from .precursor_mass_accuracy import MassAccuracyMixin
from .simple_score import SignatureAwareCoverageScorer


class LogIntensityScorer(SignatureAwareCoverageScorer, MassAccuracyMixin):

    _peptide_score = None
    _glycan_score = None

    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(LogIntensityScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)

    def calculate_score(self, error_tolerance=2e-5, peptide_weight=0.65, *args, **kwargs) -> float:
        glycan_weight = 1 - peptide_weight
        combo_score = self.peptide_score(error_tolerance, *args, **kwargs) * peptide_weight +\
                      self.glycan_score(error_tolerance, *args, **kwargs) * glycan_weight
        mass_accuracy = self._precursor_mass_accuracy_score()
        signature_component = self._signature_ion_score()
        self._score = combo_score + mass_accuracy + signature_component
        return self._score

    def calculate_peptide_score(self, error_tolerance=2e-5, peptide_coverage_weight=1.0, *args, **kwargs) -> float:
        total = 0
        series_set = {IonSeries.b, IonSeries.y, IonSeries.c, IonSeries.z}
        seen = set()
        for peak_pair in self.solution_map:
            peak = peak_pair.peak
            if peak_pair.fragment.get_series() in series_set:
                seen.add(peak.index.neutral_mass)
                total += np.log10(peak.intensity) * (1 - (abs(peak_pair.mass_accuracy()) / error_tolerance) ** 4)
        n_term, c_term = self._compute_coverage_vectors()[:2]
        coverage_score = ((n_term + c_term[::-1])).sum() / float((2 * len(self.target) - 1))
        score = total * coverage_score ** peptide_coverage_weight
        if np.isnan(score):
            return 0
        return score

    def calculate_glycan_score(self, error_tolerance=2e-5, glycan_core_weight=0.4, glycan_coverage_weight=0.5,
                               fragile_fucose=False, extended_glycan_search=False, *args, **kwargs) -> float:
        seen = set()
        series = IonSeries.stub_glycopeptide
        if not extended_glycan_search:
            theoretical_set = list(self.target.stub_fragments(extended=True))
        else:
            theoretical_set = list(self.target.stub_fragments(extended=True, extended_fucosylation=True))
        core_fragments = set()
        for frag in theoretical_set:
            if not frag.is_extended:
                core_fragments.add(frag.name)

        total = 0
        core_matches = set()
        extended_matches = set()

        for peak_pair in self.solution_map:
            if peak_pair.fragment.series != series:
                continue
            fragment_name = peak_pair.fragment.base_name()
            if fragment_name in core_fragments:
                core_matches.add(fragment_name)
            else:
                extended_matches.add(fragment_name)
            peak = peak_pair.peak
            if peak.index.neutral_mass not in seen:
                seen.add(peak.index.neutral_mass)
                total += np.log10(peak.intensity) * (1 - (abs(peak_pair.mass_accuracy()) / error_tolerance) ** 4)
        glycan_composition = self.target.glycan_composition
        n = self._get_internal_size(glycan_composition)
        k = 2.0
        if not fragile_fucose:
            side_group_count = self._glycan_side_group_count(
                glycan_composition)
            if side_group_count > 0:
                k = 1.0
        d = max(n * np.log(n) / k, n)
        core_coverage = ((len(core_matches) * 1.0) / len(core_fragments)) ** glycan_core_weight
        extended_coverage = min(float(len(core_matches) + len(extended_matches)) / d, 1.0) ** glycan_coverage_weight
        score = total * core_coverage * extended_coverage
        self._glycan_coverage = core_coverage * extended_coverage
        glycan_prior = self.target.glycan_prior
        score += self._glycan_coverage * glycan_prior
        if np.isnan(score):
            return 0
        return score

    def peptide_score(self, error_tolerance=2e-5, peptide_coverage_weight=1.0, *args, **kwargs) -> float:
        if self._peptide_score is None:
            self._peptide_score = self.calculate_peptide_score(error_tolerance, peptide_coverage_weight, *args, **kwargs)
        return self._peptide_score

    def glycan_score(self, error_tolerance=2e-5, glycan_core_weight=0.4, glycan_coverage_weight=0.5,
                     *args, **kwargs) -> float:
        if self._glycan_score is None:
            self._glycan_score = self.calculate_glycan_score(
                error_tolerance, glycan_core_weight, glycan_coverage_weight, *args, **kwargs)
        return self._glycan_score


try:
    from glycresoft._c.tandem.tandem_scoring_helpers import calculate_peptide_score, calculate_glycan_score
    LogIntensityScorer.calculate_peptide_score = calculate_peptide_score
    LogIntensityScorer.calculate_glycan_score = calculate_glycan_score
except ImportError:
    pass


class LogIntensityScorerReweighted(LogIntensityScorer):
    def peptide_score(self, error_tolerance=2e-5, peptide_coverage_weight=0.7, *args, **kwargs):
        if self._peptide_score is None:
            self._peptide_score = self.calculate_peptide_score(
                error_tolerance, peptide_coverage_weight, *args, **kwargs)
        return self._peptide_score

    def peptide_coverage(self):
        return self._calculate_peptide_coverage_no_glycosylated()

try:
    from glycresoft._c.tandem.tandem_scoring_helpers import calculate_peptide_score_no_glycosylated
    LogIntensityScorerReweighted.calculate_peptide_score = calculate_peptide_score_no_glycosylated
except ImportError:
    pass


class ShortPeptideLogIntensityScorer(LogIntensityScorer):
    stub_weight = 0.65


def _short_peptide_test(scan, target, *args, **kwargs):
    return len(target) < 10


LogIntensityModelTree = ModelTreeNode(LogIntensityScorer, {
    _short_peptide_test: ModelTreeNode(ShortPeptideLogIntensityScorer, {}),
})


class FullSignaturePenalizedLogIntensityScorer(LogIntensityScorer):
    def calculate_glycan_score(self, error_tolerance=2e-5, core_weight=0.4, coverage_weight=0.5,
                               fragile_fucose=True, extended_glycan_search=False, *args, **kwargs):
        score = super(FullSignaturePenalizedLogIntensityScorer, self).calculate_glycan_score(
            error_tolerance, core_weight, coverage_weight, fragile_fucose=fragile_fucose,
            extended_glycan_search=extended_glycan_search, *args, **kwargs)
        signature_component = self._signature_ion_score()
        return score + signature_component


class HyperscoreScorer(SignatureAwareCoverageScorer, MassAccuracyMixin):

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
        mass_accuracy = self._precursor_mass_accuracy_score()
        signature_component = self._signature_ion_score()
        self._score = hyperscore + mass_accuracy + signature_component

        return self._score
