import itertools
from operator import attrgetter
import numpy as np

from glypy.structure.glycan_composition import (
    FrozenMonosaccharideResidue,
    Composition, from_iupac_lite)

from glycopeptidepy.structure.glycan import GlycanCompositionProxy

from glycan_profiling.structure import SpectrumGraph

from glycan_profiling.tandem.oxonium_ions import SignatureSpecification, single_signatures, compound_signatures

from .base import GlycopeptideSpectrumMatcherBase


_WATER = Composition("H2O")


keyfn = attrgetter("intensity")


def base_peak_tuple(peaks):
    if peaks:
        return max(peaks, key=keyfn)
    else:
        return None


try:
    from glycan_profiling._c.tandem.tandem_scoring_helpers import base_peak_tuple
except ImportError:
    pass


class GlycanCompositionSignatureMatcher(GlycopeptideSpectrumMatcherBase):
    minimum_intensity_ratio_threshold = 0.025

    def __init__(self, scan, target, mass_shift=None):
        super(GlycanCompositionSignatureMatcher, self).__init__(scan, target, mass_shift)
        self._init_signature_matcher()

    def _init_signature_matcher(self):
        self.glycan_composition = self._copy_glycan_composition()
        self.expected_matches = dict()
        self.unexpected_matches = dict()
        self.maximum_intensity = float('inf')

    def _copy_glycan_composition(self):
        return GlycanCompositionProxy(self.target.glycan_composition)

    signatures = single_signatures
    compound_signatures = compound_signatures
    all_signatures = single_signatures.copy()
    all_signatures.update(compound_signatures)

    def match(self, error_tolerance=2e-5, rare_signatures=False, *args, **kwargs):
        if len(self.spectrum) == 0:
            return
        self.maximum_intensity = self.base_peak()
        spectrum = self.spectrum

        for mono in self.signatures:
            is_expected = mono.is_expected(self.glycan_composition)
            peak = ()
            for mass in mono.masses:
                peak += spectrum.all_peaks_for(mass, error_tolerance)
            if peak:
                peak = base_peak_tuple(peak)
            else:
                if is_expected:
                    self.expected_matches[mono] = None
                continue
            if is_expected:
                self.expected_matches[mono] = peak
            else:
                self.unexpected_matches[mono] = peak

        if rare_signatures:
            for compound in self.compound_signatures:
                is_expected = compound.is_expected(self.glycan_composition)
                peak = ()
                for mass in compound.masses:
                    peak += spectrum.all_peaks_for(mass, error_tolerance)
                if peak:
                    peak = base_peak_tuple(peak)
                else:
                    if is_expected:
                        self.expected_matches[compound] = None
                    continue
                if is_expected:
                    self.expected_matches[compound] = peak
                else:
                    self.unexpected_matches[compound] = peak

    def _find_peak_pairs(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        peak_set = self.spectrum
        if len(peak_set) == 0:
            return []
        pairs = SpectrumGraph()
        minimum_intensity_threshold = 0.01
        blocks = [(part, part.mass()) for part in self.glycan_composition]
        if include_compound:
            compound_blocks = list(itertools.combinations(self.target, 2))
            compound_blocks = [(block, sum(part.mass() for part in block))
                               for block in compound_blocks]
            blocks.extend(compound_blocks)

        max_peak = self.maximum_intensity
        threshold = max_peak * minimum_intensity_threshold

        for peak in peak_set:
            if peak.intensity < threshold or peak.neutral_mass < 150:
                continue
            for block, mass in blocks:
                for other in peak_set.all_peaks_for(peak.neutral_mass + mass, error_tolerance):
                    if other.intensity < threshold:
                        continue
                    pairs.add(peak, other, block)
        self.spectrum_graph = pairs
        return pairs

    def estimate_missing_ion_importance(self, key):
        count = key.count_of(self.glycan_composition)
        weight = self.all_signatures[key]
        return min(weight * count, 0.99)

    def _signature_ion_score(self):
        penalty = 0.0
        for key, peak in self.unexpected_matches.items():
            ratio = peak.intensity / self.maximum_intensity
            if ratio < self.minimum_intensity_ratio_threshold:
                continue
            x = 1 - ratio
            if x <= 0:
                component = 20
            else:
                component = -10 * np.log10(x)
                if np.isnan(component):
                    component = 20
            penalty += component
        for key, peak in self.expected_matches.items():
            matched = peak is not None
            if matched:
                ratio = peak.intensity / self.maximum_intensity
                if ratio < self.minimum_intensity_ratio_threshold:
                    matched = False
            if not matched:
                importance = self.estimate_missing_ion_importance(key)
                x = 1 - importance
                if x <= 0:
                    component = 20
                else:
                    component = -10 * np.log10(x)
                    if np.isnan(component):
                        component = 20
                penalty += component
        return -penalty

    def calculate_score(self, *args, **kwargs):
        self._score = self._signature_ion_score()
        return self._score

    def oxonium_ion_utilization(self):
        utilization = 0.0
        for signature, matched in self.expected_matches.items():
            if matched is None:
                comp = 10 * np.log10(1 - self.estimate_missing_ion_importance(signature))
                if np.isnan(comp):
                    comp = -20
                utilization += comp
        return utilization
