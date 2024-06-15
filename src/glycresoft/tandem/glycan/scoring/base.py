import glypy

from glycresoft.structure import FragmentMatchMap
from glycresoft.tandem.spectrum_match import SpectrumMatcherBase

water = glypy.Composition("H2O")
water_mass = water.mass


class GlycanSpectrumMatcherBase(SpectrumMatcherBase):

    def _match_fragments(self, series, error_tolerance=2e-5, max_cleavages=2, include_neutral_losses=None):
        for fragment in self.target.fragments(series, max_cleavages=max_cleavages):
            peaks = self.spectrum.all_peaks_for(fragment.mass, error_tolerance)
            for peak in peaks:
                self.solution_map.add(peak, fragment)
            if include_neutral_losses:
                for loss, label in include_neutral_losses:
                    peaks = self.spectrum.all_peaks_for(
                        fragment.mass - loss.mass, error_tolerance)
                    if peaks:
                        fragment_loss = fragment.copy()
                        fragment_loss.mass -= loss.mass
                        fragment_loss.composition -= loss
                        fragment_loss.name += label
                    for peak in peaks:
                        self.solution_map.add(peak, fragment_loss)

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        self.solution_map = FragmentMatchMap()
        include_neutral_losses = kwargs.get("include_neutral_losses", False)
        max_cleavages = kwargs.get("max_cleavages", 2)
        is_hcd = self.is_hcd()
        is_exd = self.is_exd()
        if include_neutral_losses and isinstance(include_neutral_losses, (int, bool)):
            include_neutral_losses = [(glypy.Composition("H2O"), "-H2O")]

        if is_hcd:
            self._match_fragments(
                "BY", error_tolerance, max_cleavages=max_cleavages,
                include_neutral_losses=include_neutral_losses)
        else:
            self._match_fragments(
                "ABCXYZ", error_tolerance, max_cleavages=max_cleavages,
                include_neutral_losses=include_neutral_losses)

    def _theoretical_mass(self):
        return self.target.mass()
