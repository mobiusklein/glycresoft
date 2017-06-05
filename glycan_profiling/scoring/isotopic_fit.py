import numpy as np

from ms_deisotope.scoring import g_test_scaled
from ms_deisotope.averagine import glycan, PROTON, mass_charge_ratio
from ms_peak_picker.peak_set import FittedPeak
from brainpy import isotopic_variants

from glycan_profiling.chromatogram_tree import Unmodified

from .base import ScoringFeatureBase, epsilon


def envelope_to_peak_list(envelope):
    return [FittedPeak(e[0], e[1], 0, 0, 0, 0, 0, 0, 0) for e in envelope]


def scale_theoretical_isotopic_pattern(eid, tid):
    total = sum(p.intensity for p in eid)
    for p in tid:
        p.intensity *= total


def get_nearest_index(query_mz, peak_list):
    best_index = None
    best_error = float('inf')

    for i, peak in enumerate(peak_list):
        error = abs(peak.mz - query_mz)
        if error < best_error:
            best_error = error
            best_index = i
    return best_index


def align_peak_list(experimental, theoretical):
    retain = []
    for peak in experimental:
        retain.append(theoretical[get_nearest_index(peak.mz, theoretical)])
    return retain


def unspool_nodes(node):
    yield node
    for child in (node).children:
        for x in unspool_nodes(child):
            yield x


class IsotopicPatternConsistencyFitter(ScoringFeatureBase):
    feature_type = "isotopic_fit"

    def __init__(self, chromatogram, averagine=glycan, charge_carrier=PROTON):
        self.chromatogram = chromatogram
        self.averagine = averagine
        self.charge_carrier = charge_carrier
        self.scores = []
        self.intensity = []
        self.mean_fit = None

        if chromatogram.composition is not None:
            if chromatogram.elemental_composition is not None:
                self.composition = chromatogram.elemental_composition
            else:
                raise Exception(chromatogram.composition)
        else:
            self.composition = None

        self.fit()

    def __repr__(self):
        return "IsotopicPatternConsistencyFitter(%s, %0.4f)" % (self.chromatogram, self.mean_fit)

    def generate_isotopic_pattern(self, charge, node_type=Unmodified):
        if self.composition is not None:
            tid = isotopic_variants(
                self.composition + node_type.composition,
                charge=charge, charge_carrier=self.charge_carrier)
            out = []
            total = 0.
            for p in tid:
                out.append(p)
                total += p.intensity
                if total >= 0.95:
                    break
            return out
        else:
            tid = self.averagine.isotopic_cluster(
                mass_charge_ratio(
                    self.chromatogram.neutral_mass + node_type.mass,
                    charge, charge_carrier=self.charge_carrier),
                charge,
                charge_carrier=self.charge_carrier)
            return tid

    def score_isotopic_pattern(self, deconvoluted_peak, node_type=Unmodified):
        eid, tid = self.prepare_isotopic_patterns(deconvoluted_peak, node_type)
        return g_test_scaled(None, eid, tid)

    def prepare_isotopic_patterns(self, deconvoluted_peak, node_type=Unmodified):
        tid = self.generate_isotopic_pattern(deconvoluted_peak.charge, node_type)
        eid = envelope_to_peak_list(deconvoluted_peak.envelope)
        scale_theoretical_isotopic_pattern(eid, tid)
        tid = align_peak_list(eid, tid)
        return eid, tid

    def fit(self):
        for node in self.chromatogram.nodes.unspool():
            for peak in node.members:
                score = self.score_isotopic_pattern(peak, node.node_type)
                self.scores.append(score)
                self.intensity.append(peak.intensity)
        self.intensity = np.array(self.intensity)
        self.scores = np.array(self.scores)
        self.mean_fit = np.average(self.scores, weights=self.intensity)

    @classmethod
    def score(cls, chromatogram, *args, **kwargs):
        return max(1 - cls(chromatogram).mean_fit, epsilon)
