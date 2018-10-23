import numpy as np
import math

from glycan_profiling.structure import FragmentMatchMap
from glycopeptidepy.structure.fragment import IonSeries
from glycopeptidepy.structure.fragmentation_strategy import EXDFragmentationStrategy, HCDFragmentationStrategy

from ms_deisotope.data_source.metadata import activation


from .base import ChemicalShift, ModelTreeNode
from .simple_score import SimpleCoverageScorer
from .precursor_mass_accuracy import MassAccuracyModel
from .glycan_signature_ions import GlycanCompositionSignatureMatcher


class LogIntensityScorer(SimpleCoverageScorer, GlycanCompositionSignatureMatcher):
    accuracy_bias = MassAccuracyModel(-2.673807e-07, 5.022458e-06)

    def __init__(self, scan, sequence, mass_shift=None, *args, **kwargs):
        super(LogIntensityScorer, self).__init__(scan, sequence, mass_shift, *args, **kwargs)

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None):
        if strategy is None:
            strategy = EXDFragmentationStrategy
        for frags in self.target.get_fragments(series, strategy=strategy):
            glycosylated_position = False
            for frag in frags:
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

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        GlycanCompositionSignatureMatcher.match(self, error_tolerance=error_tolerance)
        masked_peaks = set()

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
        return self

    def _intensity_score(self, error_tolerance=2e-5, *args, **kwargs):
        total = 0
        for peak, fragment in self.solution_map:
            total += np.log10(peak.intensity)
        return total

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        coverage_score = self._coverage_score(*args, **kwargs)
        intensity_score = self._intensity_score(error_tolerance, *args, **kwargs)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = GlycanCompositionSignatureMatcher.calculate_score(self)
        self._score = intensity_score * coverage_score + mass_accuracy + signature_component

        return self._score
