from operator import attrgetter

import numpy as np

from .base import GlycopeptideSpectrumMatcherBase

from glypy.composition.glycan_composition import (
    FrozenGlycanComposition, FrozenMonosaccharideResidue,
    MonosaccharideResidue, Composition)


signatures = {
    FrozenMonosaccharideResidue.from_iupac_lite("NeuAc"): 0.5,
    FrozenMonosaccharideResidue.from_iupac_lite("NeuGc"): 0.5,
}

_water = Composition("H2O")


class GlycanCompositionSignatureMatcher(GlycopeptideSpectrumMatcherBase):
    def __init__(self, scan, target, mass_shift=None):
        super(GlycanCompositionSignatureMatcher, self).__init__(scan, target, mass_shift)
        self.glycan_composition = self._copy_glycan_composition()
        self.expected_matches = dict()
        self.unexpected_matches = dict()
        self.maximum_intensity = float('inf')

    def _copy_glycan_composition(self):
        return FrozenGlycanComposition(self.target.glycan_composition)

    signatures = signatures

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        try:
            self.maximum_intensity = max([peak.intensity for peak in self.spectrum])
        except ValueError:
            return
        water = _water
        spectrum = self.spectrum
        keyfn = attrgetter("intensity")

        for monosaccharide in self.signatures:
            is_expected = monosaccharide in self.glycan_composition
            peak = spectrum.all_peaks_for(monosaccharide.mass(), error_tolerance)
            peak += spectrum.all_peaks_for(monosaccharide.mass() - water.mass, error_tolerance)
            if peak:
                peak = max(peak, key=keyfn)
            else:
                if is_expected:
                    self.expected_matches[monosaccharide] = None
                continue
            if is_expected:
                self.expected_matches[monosaccharide] = peak
            else:
                self.unexpected_matches[monosaccharide] = peak
        
    def estimate_missing_ion_importance(self, key):
        count = self.glycan_composition[key]
        weight = self.signatures[key]
        return min(weight * count, 0.99)

    def calculate_score(self, match_tolerance=2e-5, *args, **kwargs):
        penalty = 0
        for key, peak in self.unexpected_matches.items():
            ratio = peak.intensity / self.maximum_intensity
            if ratio < 0.01:
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
            if peak is None:
                importance = self.estimate_missing_ion_importance(key)
                x = 1 - importance
                if x <= 0:
                    component = 20
                else:
                    component = -10 * np.log10(x)
                    if np.isnan(component):
                        component = 20
                penalty += component
                
        self._score = -penalty
        return self._score
