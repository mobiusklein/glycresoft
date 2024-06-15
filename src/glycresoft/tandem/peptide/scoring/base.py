from glycopeptidepy.structure.fragment import ChemicalShift, IonSeries, SimpleFragment, Composition
from glycopeptidepy.structure.fragmentation_strategy import EXDFragmentationStrategy, HCDFragmentationStrategy

from ...spectrum_match import SpectrumMatcherBase


class PeptideSpectrumMatcherBase(SpectrumMatcherBase):
    def get_fragments(self, series, strategy=None, include_neutral_losses=False):
        fragments = self.target.get_fragments(
            series, strategy=strategy, include_neutral_losses=include_neutral_losses)
        return fragments

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None,
                               include_neutral_losses=False):
        if strategy is None:
            strategy = HCDFragmentationStrategy
        if masked_peaks is None:
            masked_peaks = set()
        for frags in self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses):
            for frag in frags:
                for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    self.solution_map.add(peak, frag)

    def _theoretical_mass(self):
        return self.target.total_mass

    def _match_precursor(self, error_tolerance=2e-5, masked_peaks=None, include_neutral_losses=False):
        if masked_peaks is None:
            masked_peaks = set()
        mass = self.target.total_mass
        frag = SimpleFragment("M", mass, IonSeries.precursor, None)
        for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
            if peak.index.neutral_mass in masked_peaks:
                continue
            masked_peaks.add(peak)
            self.solution_map.add(peak, frag)
        if include_neutral_losses:
            delta = -Composition("NH3")
            for peak in self.spectrum.all_peaks_for(frag.mass + delta.mass, error_tolerance):
                if peak.index.neutral_mass in masked_peaks:
                    continue
                masked_peaks.add(peak)
                shifted_frag = frag.clone()
                shifted_frag.set_chemical_shift(ChemicalShift("-NH3", delta))
                self.solution_map.add(peak, shifted_frag)
            delta = -Composition("H2O")
            for peak in self.spectrum.all_peaks_for(frag.mass + delta.mass, error_tolerance):
                if peak.index.neutral_mass in masked_peaks:
                    continue
                masked_peaks.add(peak)
                shifted_frag = frag.clone()
                shifted_frag.set_chemical_shift(ChemicalShift("-H2O", delta))
                self.solution_map.add(peak, shifted_frag)

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        masked_peaks = set()
        include_neutral_losses = kwargs.get("include_neutral_losses", False)
        is_hcd = self.is_hcd()
        is_exd = self.is_exd()
        if not is_hcd and not is_exd:
            is_hcd = True

        self._match_precursor(error_tolerance, masked_peaks, include_neutral_losses)

        # handle N-term
        if is_hcd and not is_exd:
            self._match_backbone_series(
                IonSeries.b, error_tolerance, masked_peaks, HCDFragmentationStrategy,
                include_neutral_losses)
        elif is_exd:
            self._match_backbone_series(
                IonSeries.b, error_tolerance, masked_peaks, EXDFragmentationStrategy,
                include_neutral_losses)
            self._match_backbone_series(
                IonSeries.c, error_tolerance, masked_peaks, EXDFragmentationStrategy,
                include_neutral_losses)

        # handle C-term
        if is_hcd and not is_exd:
            self._match_backbone_series(
                IonSeries.y, error_tolerance, masked_peaks, HCDFragmentationStrategy,
                include_neutral_losses)
        elif is_exd:
            self._match_backbone_series(
                IonSeries.y, error_tolerance, masked_peaks, EXDFragmentationStrategy,
                include_neutral_losses)
            self._match_backbone_series(
                IonSeries.z, error_tolerance, masked_peaks, EXDFragmentationStrategy,
                include_neutral_losses)
        return self


try:
    from glycresoft._c.tandem.tandem_scoring_helpers import (
        PeptideSpectrumMatcherBase_match_backbone_series, _match_precursor)
    PeptideSpectrumMatcherBase._match_backbone_series = PeptideSpectrumMatcherBase_match_backbone_series
    PeptideSpectrumMatcherBase._match_precursor = _match_precursor
except ImportError:
    pass
