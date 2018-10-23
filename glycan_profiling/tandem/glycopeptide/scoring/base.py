from glycopeptidepy.structure.fragment import ChemicalShift, IonSeries
from glycopeptidepy.structure.fragmentation_strategy import EXDFragmentationStrategy, HCDFragmentationStrategy

from ...spectrum_match import SpectrumMatcherBase, ModelTreeNode


class GlycopeptideSpectrumMatcherBase(SpectrumMatcherBase):

    def _match_oxonium_ions(self, error_tolerance=2e-5, masked_peaks=None):
        if masked_peaks is None:
            masked_peaks = set()
        for frag in self.target.glycan_fragments(
                all_series=False, allow_ambiguous=False,
                include_large_glycan_fragments=False,
                maximum_fragment_size=4):
            peak = self.spectrum.has_peak(frag.mass, error_tolerance)
            if peak and peak.index.neutral_mass not in masked_peaks:
                self.solution_map.add(peak, frag)
                masked_peaks.add(peak.index.neutral_mass)
        return masked_peaks

    def _match_stub_glycopeptides(self, error_tolerance=2e-5, masked_peaks=None, chemical_shift=None):
        if masked_peaks is None:
            masked_peaks = set()
        for frag in self.target.stub_fragments(extended=True):
            for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
                # should we be masking these? peptides which have amino acids which are
                # approximately the same mass as a monosaccharide unit at ther terminus
                # can produce cases where a stub ion and a backbone fragment match the
                # same peak.
                #
                masked_peaks.add(peak.index.neutral_mass)
                self.solution_map.add(peak, frag)
                if chemical_shift is not None:
                    shifted_mass = frag.mass + self.mass_shift.tandem_mass
                    for peak in self.spectrum.all_peaks_for(shifted_mass, error_tolerance):
                        masked_peaks.add(peak.index.neutral_mass)
                        shifted_frag = frag.clone()
                        shifted_frag.chemical_shift = chemical_shift
                        shifted_frag.name += "+ %s" % (self.mass_shift.name,)
                        self.solution_map.add(peak, shifted_frag)
        return masked_peaks

    def get_fragments(self, series, strategy=None):
        fragments = self.target.get_fragments(series, strategy=strategy)
        return fragments

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None):
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
