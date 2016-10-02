import math

from ...spectrum_matcher_base import SpectrumMatcherBase

from glycresoft_sqlalchemy.structure.fragment import IonSeries


def is_glycosylated(frag):
    return "HexNAc" in frag.modification_dict


class SimpleCoverageScorer(SpectrumMatcherBase):
    def __init__(self, scan, sequence):
        super(SimpleCoverageScorer, self).__init__(scan, sequence)
        self._score = None
        self.solution_map = dict()
        self.glycosylated_b_ion_count = 0
        self.glycosylated_y_ion_count = 0

    def match(self, error_tolerance=2e-5):
        solution_map = dict()
        spectrum = self.spectrum

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= is_glycosylated(frag)
                peak = spectrum.has_peak(frag.mass, error_tolerance)
                if peak:
                    solution_map[frag] = peak
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= is_glycosylated(frag)
                peak = spectrum.has_peak(frag.mass, error_tolerance)
                if peak:
                    solution_map[frag] = peak
            if glycosylated_position:
                n_glycosylated_y_ions += 1
        for frag in self.target.stub_fragments(extended=True):
            peak = spectrum.has_peak(frag.mass, error_tolerance)
            if peak:
                solution_map[frag] = peak

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

        for frag in self.solution_map:
            if frag.series == IonSeries.b:
                b_ions[frag.position] = 1
                if is_glycosylated(frag):
                    glycosylated_b_ions += 1
            elif frag.series == IonSeries.y:
                y_ions[frag.position] = 1
                if is_glycosylated(frag):
                    glycosylated_y_ions += 1
            elif frag.series == IonSeries.stub_glycopeptide:
                stub_count += 1

        mean_coverage = sum([math.log(1 + (b + y), 2) / math.log(3, 2)
                             for b, y in zip(b_ions, y_ions[::-1])]) / float(len(self.target))

        glycosylated_coverage = (
            glycosylated_b_ions + glycosylated_y_ions) / float(
            self.glycosylated_b_ion_count + self.glycosylated_y_ion_count)

        stub_fraction = min(stub_count, 3) / 3.

        return mean_coverage, glycosylated_coverage, stub_fraction

    def compute_score(self, backbone_weight=0.5, hexnac_weight=0.5, stub_weight=0.2):
        mean_coverage, glycosylated_coverage, stub_fraction = self.compute_coverage()
        score = (((mean_coverage * backbone_weight) + (glycosylated_coverage * hexnac_weight)) * (
            1 - stub_weight)) + (stub_fraction * stub_weight)
        self._score = score
        return score
