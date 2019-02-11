from glycopeptidepy.structure.fragment import ChemicalShift, IonSeries
from glycopeptidepy.structure.fragmentation_strategy import EXDFragmentationStrategy, HCDFragmentationStrategy

from ...spectrum_match import SpectrumMatcherBase


class PeptideSpectrumMatcherBase(SpectrumMatcherBase):
    def get_fragments(self, series, strategy=None):
        fragments = self.target.get_fragments(series, strategy=strategy)
        return fragments

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None,
                               track_ions=True, **kwargs):
        if strategy is None:
            strategy = HCDFragmentationStrategy
        if masked_peaks is None:
            masked_peaks = set()
        for frags in self.target.get_fragments(series, strategy=strategy):
            for frag in frags:
                for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    self.solution_map.add(peak, frag)

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        masked_peaks = set()

        is_hcd = self.is_hcd()
        is_exd = self.is_exd()

        # handle N-term
        if is_hcd and not is_exd:
            self._match_backbone_series(IonSeries.b, error_tolerance, masked_peaks, HCDFragmentationStrategy)
        elif is_exd:
            self._match_backbone_series(IonSeries.b, error_tolerance, masked_peaks, EXDFragmentationStrategy)
            self._match_backbone_series(IonSeries.c, error_tolerance, masked_peaks, EXDFragmentationStrategy)

        # handle C-term
        if is_hcd and not is_exd:
            self._match_backbone_series(IonSeries.y, error_tolerance, masked_peaks, HCDFragmentationStrategy)
        elif is_exd:
            self._match_backbone_series(IonSeries.y, error_tolerance, masked_peaks, EXDFragmentationStrategy)
            self._match_backbone_series(IonSeries.z, error_tolerance, masked_peaks, EXDFragmentationStrategy)
            self._match_backbone_series(
                IonSeries.zp, error_tolerance, masked_peaks, EXDFragmentationStrategy, track_ions=False)
        return self
