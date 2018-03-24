import math

from .base import ChemicalShift, ModelTreeNode
from .binomial_score import BinomialSpectrumMatcher
from .simple_score import SimpleCoverageScorer
from .precursor_mass_accuracy import MassAccuracyModel
from .glycan_signature_ions import GlycanCompositionSignatureMatcher
from glycan_profiling.structure import FragmentMatchMap


# This probably shouldn't be global
accuracy_bias = MassAccuracyModel(-2.673807e-07, 5.022458e-06)


class CoverageWeightedBinomialScorer(BinomialSpectrumMatcher, SimpleCoverageScorer, GlycanCompositionSignatureMatcher):
    accuracy_bias = accuracy_bias

    def __init__(self, scan, sequence, mass_shift=None):
        BinomialSpectrumMatcher.__init__(self, scan, sequence, mass_shift)
        self.glycosylated_b_ion_count = 0
        self.glycosylated_y_ion_count = 0

        self.glycan_composition = self._copy_glycan_composition()
        self.expected_matches = dict()
        self.unexpected_matches = dict()
        self.maximum_intensity = float('inf')

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        GlycanCompositionSignatureMatcher.match(self, error_tolerance=error_tolerance)
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
        n_theoretical = 0
        backbone_mass_series = []
        neutral_losses = tuple(kwargs.pop("neutral_losses", []))

        masked_peaks = set()
        for frag in self.target.glycan_fragments(
                all_series=False, allow_ambiguous=False,
                include_large_glycan_fragments=False,
                maximum_fragment_size=4):
            peak = spectrum.has_peak(frag.mass, error_tolerance)
            if peak:
                solution_map.add(peak, frag)
                masked_peaks.add(peak.index.neutral_mass)
                try:
                    self._sanitized_spectrum.remove(peak)
                except KeyError:
                    continue
        if self.mass_shift.tandem_mass != 0:
            chemical_shift = ChemicalShift(
                self.mass_shift.name, self.mass_shift.tandem_composition)
        else:
            chemical_shift = None
        for frag in self.target.stub_fragments(extended=True):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                # should we be masking these? peptides which have amino acids which are
                # approximately the same mass as a monosaccharide unit at ther terminus
                # can produce cases where a stub ion and a backbone fragment match the
                # same peak.
                #
                masked_peaks.add(peak.index.neutral_mass)
                solution_map.add(peak, frag)

            # If the precursor match was caused by a mass shift, that mass shift may
            # be associated with stub fragments.
            if chemical_shift is not None:
                shifted_mass = frag.mass + self.mass_shift.tandem_mass
                for peak in spectrum.all_peaks_for(shifted_mass, error_tolerance):
                    masked_peaks.add(peak.index.neutral_mass)
                    shifted_frag = frag.clone()
                    shifted_frag.chemical_shift = chemical_shift
                    shifted_frag.name += "+ %s" % (self.mass_shift.name,)
                    solution_map.add(peak, shifted_frag)

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b', neutral_losses):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                backbone_mass_series.append(frag)
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y', neutral_losses):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                backbone_mass_series.append(frag)
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_y_ions += 1

        self.n_theoretical = n_theoretical
        self.glycosylated_b_ion_count = n_glycosylated_b_ions
        self.glycosylated_y_ion_count = n_glycosylated_y_ions
        self.solution_map = solution_map
        self._backbone_mass_series = backbone_mass_series
        return solution_map

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None, glycosylated_weight=None,
                        stub_weight=None, *args, **kwargs):
        bin_score = BinomialSpectrumMatcher.calculate_score(
            self, error_tolerance=error_tolerance)
        coverage_score = SimpleCoverageScorer.calculate_score(
            self, backbone_weight, glycosylated_weight, stub_weight)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = GlycanCompositionSignatureMatcher.calculate_score(self)
        self._score = bin_score * coverage_score + mass_accuracy + signature_component
        return self._score


class ShortPeptideCoverageWeightedBinomialScorer(CoverageWeightedBinomialScorer):
    stub_weight = 0.75


def _short_peptide_test(scan, target, *args, **kwargs):
    return len(target) < 10


CoverageWeightedBinomialModelTree = ModelTreeNode(CoverageWeightedBinomialScorer, {
    _short_peptide_test: ModelTreeNode(ShortPeptideCoverageWeightedBinomialScorer, {}),
})
