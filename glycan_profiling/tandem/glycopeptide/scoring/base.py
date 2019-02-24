from glycopeptidepy.structure.fragment import ChemicalShift, IonSeries
from glycopeptidepy.structure.fragmentation_strategy import EXDFragmentationStrategy, HCDFragmentationStrategy

from ...spectrum_match import SpectrumMatcherBase, ModelTreeNode


class GlycopeptideSpectrumMatcherBase(SpectrumMatcherBase):

    _glycan_score = None
    _peptide_score = None

    def _theoretical_mass(self):
        return self.target.total_mass

    def _match_oxonium_ions(self, error_tolerance=2e-5, masked_peaks=None):
        '''Note: This method is masked by a Cython implementation
        '''
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

    def get_fragments(self, series, strategy=None, **kwargs):
        fragments = self.target.get_fragments(series, strategy=strategy)
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

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        masked_peaks = set()
        include_neutral_losses = kwargs.get("include_neutral_losses", False)
        if self.mass_shift.tandem_mass != 0:
            chemical_shift = ChemicalShift(
                self.mass_shift.name, self.mass_shift.tandem_composition)
        else:
            chemical_shift = None

        is_hcd = self.is_hcd()
        is_exd = self.is_exd()

        # handle glycan fragments from collisional dissociation
        if is_hcd:
            self._match_oxonium_ions(error_tolerance, masked_peaks=masked_peaks)
            self._match_stub_glycopeptides(error_tolerance, masked_peaks=masked_peaks, chemical_shift=chemical_shift)

        # handle N-term
        if is_hcd and not is_exd:
            self._match_backbone_series(
                IonSeries.b, error_tolerance, masked_peaks, HCDFragmentationStrategy, include_neutral_losses)
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

    def peptide_score(self, *args, **kwargs):
        if self._peptide_score is None:
            self._peptide_score = self.calculate_peptide_score(*args, **kwargs)
        return self._peptide_score

    def calculate_peptide_score(self, *args, **kwargs):
        return 0

    def glycan_score(self, *args, **kwargs):
        if self._glycan_score is None:
            self._glycan_score = self.calculate_glycan_score(*args, **kwargs)
        return self._glycan_score

    def calculate_glycan_score(self, *args, **kwargs):
        return 0


try:
    from glycan_profiling._c.tandem.tandem_scoring_helpers import _match_oxonium_ions, _match_stub_glycopeptides
    GlycopeptideSpectrumMatcherBase._match_oxonium_ions = _match_oxonium_ions
    GlycopeptideSpectrumMatcherBase._match_stub_glycopeptides = _match_stub_glycopeptides
except ImportError:
    pass
