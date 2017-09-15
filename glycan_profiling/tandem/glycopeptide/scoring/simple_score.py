import math

from .base import GlycopeptideSpectrumMatcherBase
from glycan_profiling.structure import FragmentMatchMap

from glycopeptidepy.structure.fragment import IonSeries


class SimpleCoverageScorer(GlycopeptideSpectrumMatcherBase):
    def __init__(self, scan, sequence):
        super(SimpleCoverageScorer, self).__init__(scan, sequence)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.glycosylated_b_ion_count = 0
        self.glycosylated_y_ion_count = 0

    def match(self, error_tolerance=2e-5):
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_y_ions += 1
        for frag in self.target.stub_fragments(extended=True):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                solution_map.add(peak, frag)

        self.glycosylated_b_ion_count = n_glycosylated_b_ions
        self.glycosylated_y_ion_count = n_glycosylated_y_ions
        self.solution_map = solution_map
        return solution_map

    def compute_coverage(self):
        b_ions = [0] * len(self.target)
        y_ions = [0] * len(self.target)
        stub_count = 0
        glycosylated_b_ions = 0
        glycosylated_y_ions = 0

        for frag in self.solution_map.fragments():
            if frag.series == IonSeries.b:
                b_ions[frag.position] = 1
                if frag.is_glycosylated:
                    glycosylated_b_ions += 1
            elif frag.series == IonSeries.y:
                y_ions[frag.position] = 1
                if frag.is_glycosylated:
                    glycosylated_y_ions += 1
            elif frag.series == IonSeries.stub_glycopeptide:
                stub_count += 1

        mean_coverage = sum([math.log(1 + (b + y), 2) / math.log(3, 2)
                             for b, y in zip(b_ions, y_ions[::-1])]) / float(len(self.target))

        glycosylated_coverage = 0.
        ladders = 0.
        if self.glycosylated_b_ion_count > 0:
            glycosylated_coverage += (glycosylated_b_ions / float(self.glycosylated_b_ion_count))
            ladders += 1.
        if self.glycosylated_y_ion_count > 0:
            glycosylated_coverage += (glycosylated_y_ions / float(self.glycosylated_y_ion_count))
            ladders += 1.
        if ladders > 0:
            glycosylated_coverage /= ladders

        stub_fraction = min(stub_count, 3) / 3.

        return mean_coverage, glycosylated_coverage, stub_fraction

    def calculate_score(self, backbone_weight=0.5, glycosylated_weight=0.5, stub_weight=0.2, **kwargs):
        score = self._coverage_score(backbone_weight, glycosylated_weight, stub_weight)
        self._score = score
        return score

    def _coverage_score(self, backbone_weight=0.5, glycosylated_weight=0.5, stub_weight=0.2):
        mean_coverage, glycosylated_coverage, stub_fraction = self.compute_coverage()
        score = (((mean_coverage * backbone_weight) + (glycosylated_coverage * glycosylated_weight)) * (
            1 - stub_weight)) + (stub_fraction * stub_weight)
        return score
