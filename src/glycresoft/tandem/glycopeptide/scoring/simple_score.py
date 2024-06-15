import numpy as np

from glycresoft.structure import FragmentMatchMap

from glycresoft.tandem.glycopeptide.core_search import approximate_internal_size_of_glycan

from .base import (
    GlycopeptideSpectrumMatcherBase, ChemicalShift, EXDFragmentationStrategy,
    HCDFragmentationStrategy, IonSeries)

from .glycan_signature_ions import GlycanCompositionSignatureMatcher


class SimpleCoverageScorer(GlycopeptideSpectrumMatcherBase):
    backbone_weight = 0.5
    glycosylated_weight = 0.5
    stub_weight = 0.2

    def __init__(self, scan, sequence, mass_shift=None):
        super(SimpleCoverageScorer, self).__init__(scan, sequence, mass_shift)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.glycosylated_n_term_ion_count = 0
        self.glycosylated_c_term_ion_count = 0

    @property
    def glycosylated_b_ion_count(self):
        return self.glycosylated_n_term_ion_count

    @glycosylated_b_ion_count.setter
    def glycosylated_b_ion_count(self, value):
        self.glycosylated_n_term_ion_count = value

    @property
    def glycosylated_y_ion_count(self):
        return self.glycosylated_c_term_ion_count

    @glycosylated_y_ion_count.setter
    def glycosylated_y_ion_count(self, value):
        self.glycosylated_c_term_ion_count = value

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None,
                               include_neutral_losses=False):
        if strategy is None:
            strategy = HCDFragmentationStrategy
        # Assumes that fragmentation proceeds from the start of the ladder (series position 1)
        # which means that if the last fragment could be glycosylated then the next one will be
        # but if the last fragment wasn't the next one might be.
        previous_position_glycosylated = False
        for frags in self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses):
            glycosylated_position = previous_position_glycosylated
            for frag in frags:
                if not glycosylated_position:
                    glycosylated_position |= frag.is_glycosylated
                for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    self.solution_map.add(peak, frag)
            if glycosylated_position:
                if series.direction > 0:
                    self.glycosylated_n_term_ion_count += 1
                else:
                    self.glycosylated_c_term_ion_count += 1
            previous_position_glycosylated = glycosylated_position

    def _compute_coverage_vectors(self):
        n_term_ions = np.zeros(len(self.target))
        c_term_ions = np.zeros(len(self.target))
        stub_count = 0
        glycosylated_n_term_ions = set()
        glycosylated_c_term_ions = set()

        for frag in self.solution_map.fragments():
            series = frag.get_series()
            if series in (IonSeries.b, IonSeries.c):
                n_term_ions[frag.position] = 1
                if frag.is_glycosylated:
                    glycosylated_n_term_ions.add((series, frag.position))
            elif series in (IonSeries.y, IonSeries.z):
                c_term_ions[frag.position] = 1
                if frag.is_glycosylated:
                    glycosylated_c_term_ions.add((series, frag.position))
            elif series == IonSeries.stub_glycopeptide:
                stub_count += 1

        n_term_ions[0] = 0
        return n_term_ions, c_term_ions, stub_count, len(glycosylated_n_term_ions), len(glycosylated_c_term_ions)

    def _compute_glycosylated_coverage(self, glycosylated_n_term_ions, glycosylated_c_term_ions):
        ladders = 0.
        numer = 0.0
        denom = 0.0
        if self.glycosylated_n_term_ion_count > 0:
            numer += glycosylated_n_term_ions
            denom += self.glycosylated_n_term_ion_count
            ladders += 1.
        if self.glycosylated_c_term_ion_count > 0:
            numer += glycosylated_c_term_ions
            denom += self.glycosylated_c_term_ion_count
            ladders += 1.
        if denom == 0.0:
            return 0.0
        return numer / denom

    def _get_internal_size(self, glycan_composition):
        return approximate_internal_size_of_glycan(glycan_composition)

    def compute_coverage(self):
        (n_term_ions, c_term_ions, stub_count,
         glycosylated_n_term_ions,
         glycosylated_c_term_ions) = self._compute_coverage_vectors()

        mean_coverage = np.mean(np.log2(n_term_ions + c_term_ions[::-1] + 1) / np.log2(3))

        glycosylated_coverage = self._compute_glycosylated_coverage(
            glycosylated_n_term_ions,
            glycosylated_c_term_ions)

        stub_fraction = min(stub_count, 3) / 3.

        return mean_coverage, glycosylated_coverage, stub_fraction

    @classmethod
    def get_params(self, backbone_weight=None, glycosylated_weight=None, stub_weight=None, **kwargs):
        if backbone_weight is None:
            backbone_weight = self.backbone_weight
        if glycosylated_weight is None:
            glycosylated_weight = self.glycosylated_weight
        if stub_weight is None:
            stub_weight = self.stub_weight
        return backbone_weight, glycosylated_weight, stub_weight, kwargs

    def calculate_score(self, backbone_weight=None, glycosylated_weight=None, stub_weight=None, **kwargs):
        if backbone_weight is None:
            backbone_weight = self.backbone_weight
        if glycosylated_weight is None:
            glycosylated_weight = self.glycosylated_weight
        if stub_weight is None:
            stub_weight = self.stub_weight
        score = self._coverage_score(backbone_weight, glycosylated_weight, stub_weight)
        self._score = score
        return score

    def _coverage_score(self, backbone_weight=None, glycosylated_weight=None, stub_weight=None):
        if backbone_weight is None:
            backbone_weight = self.backbone_weight
        if glycosylated_weight is None:
            glycosylated_weight = self.glycosylated_weight
        if stub_weight is None:
            stub_weight = self.stub_weight
        mean_coverage, glycosylated_coverage, stub_fraction = self.compute_coverage()
        score = (((mean_coverage * backbone_weight) + (glycosylated_coverage * glycosylated_weight)) * (
            1 - stub_weight)) + (stub_fraction * stub_weight)
        return score


class SignatureAwareCoverageScorer(SimpleCoverageScorer, GlycanCompositionSignatureMatcher):
    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(SignatureAwareCoverageScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        GlycanCompositionSignatureMatcher.match(self, error_tolerance=error_tolerance, **kwargs)
        masked_peaks = set()
        include_neutral_losses = kwargs.get("include_neutral_losses", False)
        extended_glycan_search = kwargs.get("extended_glycan_search", False)

        if self.mass_shift.tandem_mass != 0:
            chemical_shift = ChemicalShift(
                self.mass_shift.name, self.mass_shift.tandem_composition)
        else:
            chemical_shift = None

        is_hcd = self.is_hcd()
        is_exd = self.is_exd()
        if not is_hcd and not is_exd:
            is_hcd = True
        # handle glycan fragments from collisional dissociation
        if is_hcd:
            self._match_oxonium_ions(error_tolerance, masked_peaks=masked_peaks)
            self._match_stub_glycopeptides(error_tolerance, masked_peaks=masked_peaks,
                                           chemical_shift=chemical_shift,
                                           extended_glycan_search=extended_glycan_search)
        # handle N-term
        if is_hcd and not is_exd:
            self._match_backbone_series(
                IonSeries.b, error_tolerance, masked_peaks, HCDFragmentationStrategy, include_neutral_losses)
        elif is_exd:
            self._match_backbone_series(
                IonSeries.b, error_tolerance, masked_peaks, EXDFragmentationStrategy, include_neutral_losses)
            self._match_backbone_series(
                IonSeries.c, error_tolerance, masked_peaks, EXDFragmentationStrategy, include_neutral_losses)

        # handle C-term
        if is_hcd and not is_exd:
            self._match_backbone_series(
                IonSeries.y, error_tolerance, masked_peaks, HCDFragmentationStrategy, include_neutral_losses)
        elif is_exd:
            self._match_backbone_series(
                IonSeries.y, error_tolerance, masked_peaks, EXDFragmentationStrategy, include_neutral_losses)
            self._match_backbone_series(
                IonSeries.z, error_tolerance, masked_peaks, EXDFragmentationStrategy, include_neutral_losses)
        return self


try:
    from glycresoft._c.tandem.tandem_scoring_helpers import (
        _compute_coverage_vectors, SimpleCoverageScorer_match_backbone_series)
    SimpleCoverageScorer._compute_coverage_vectors = _compute_coverage_vectors
    SimpleCoverageScorer._match_backbone_series = SimpleCoverageScorer_match_backbone_series
except ImportError:
    pass
