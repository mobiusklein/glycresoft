from collections import Counter

from itertools import chain
import math

from ...spectrum_matcher_base import SpectrumMatcherBase
from glycan_profiling.structure import FragmentMatchMap

from glycopeptidepy.structure.fragment import IonSeries


def flatten(iterable):
    return [a for b in iterable for a in b]


class FrequencyCounter(object):
    def __init__(self):
        self.observed_pairs = Counter()
        self.observed_n_term = Counter()
        self.observed_c_term = Counter()
        self.observed_sequences = Counter()

        self.possible_pairs = Counter()
        self.possible_n_term = Counter()
        self.possible_c_term = Counter()

        self.series = tuple("by")

    def add_sequence(self, glycopeptide):
        self.observed_sequences[glycopeptide] += 1

    def add_fragment(self, fragment):
        n_term, c_term = fragment.flanking_amino_acids
        self.observed_pairs[n_term, c_term] += 1
        self.observed_n_term[n_term] += 1
        self.observed_c_term[c_term] += 1

    def total_possible_outcomes(self):
        possible_pairs = Counter()
        possible_n_term = Counter()
        possible_c_term = Counter()
        for seq, count in self.observed_sequences.items():
            for series in self.series:
                for fragment in flatten(seq.get_fragments(series)):
                    n_term, c_term = fragment.flanking_amino_acids
                    possible_pairs[n_term, c_term] += count
                    possible_n_term[n_term] += count
                    possible_c_term[c_term] += count

        self.possible_pairs = possible_pairs
        self.possible_n_term = possible_n_term
        self.possible_c_term = possible_c_term

    def process_match(self, spectrum_match):
        for case in spectrum_match.solution_map:
            if case.series == 'b' or case.series == 'y':
                self.add_fragment(case)
        self.add_sequence(spectrum_match.target)

    def n_term_probability(self, residue=None):
        if residue is not None:
            return self.observed_n_term[residue] / float(self.possible_n_term[residue])
        else:
            return {r: self.n_term_probability(r) for r in self.observed_n_term}

    def c_term_probability(self, residue=None):
        if residue is not None:
            return self.observed_c_term[residue] / float(self.possible_c_term[residue])
        else:
            return {r: self.c_term_probability(r) for r in self.observed_c_term}

    def pair_probability(self, pair=None):
        if pair is not None:
            pair = tuple(pair)
            return self.observed_pairs[pair] / float(self.possible_pairs[pair])
        else:
            return {r: self.pair_probability(r) for r in self.observed_pairs}

    def residue_probability(self, residue=None):
        if residue is not None:
            return (self.observed_n_term[residue] + self.observed_c_term[residue]) / float(
                self.possible_n_term[residue] + self.possible_c_term[residue])
        else:
            return {r: self.residue_probability(r) for r in (set(self.observed_c_term) | set(self.observed_n_term))}

    @classmethod
    def fit(cls, spectrum_match_set):
        inst = cls()
        for match in spectrum_match_set:
            inst.process_match(match)
        inst.total_possible_outcomes()
        return inst


class FrequencyScorer(SpectrumMatcherBase):
    def __init__(self, scan, sequence, model=None):
        super(FrequencyScorer, self).__init__(scan, sequence)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.glycosylated_b_ion_count = 0
        self.glycosylated_y_ion_count = 0
        self.model = model

    def match(self, error_tolerance=2e-5):
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                peak = spectrum.has_peak(frag.mass, error_tolerance)
                if peak:
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y'):
            glycosylated_position = False
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                peak = spectrum.has_peak(frag.mass, error_tolerance)
                if peak:
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_y_ions += 1
        for frag in self.target.stub_fragments(extended=True):
            peak = spectrum.has_peak(frag.mass, error_tolerance)
            if peak:
                solution_map.add(peak, frag)

        self.glycosylated_b_ion_count = n_glycosylated_b_ions
        self.glycosylated_y_ion_count = n_glycosylated_y_ions
        self.solution_map = solution_map
        return solution_map

    def _compute_total(self):
        total = 0.
        for frags in chain(self.target.get_fragments('b'), self.target.get_fragments('y')):
            for frag in frags:
                n_term, c_term = frag.flanking_amino_acids
                score = self.frequency_counter.n_term_probability(
                    n_term) * self.frequency_counter.c_term_probability(c_term)
                total += score * 0.5
        return total

    def _score_backbone(self):
        total = self._compute_total()
        observed = 0.0
        track_site = set()
        for frag in self.solution_map.fragments():
            if (frag.series == 'b') or (frag.series == 'y'):
                position = frag.position
                n_term, c_term = frag.flanking_amino_acids
                score = self.model.n_term_probability(
                    n_term) * self.model.c_term_probability(c_term)
                weight = 0.6 if position not in track_site else 0.4
                track_site.add(position)
                observed += score * weight
        return observed / total
