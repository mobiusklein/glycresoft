import math
import itertools

from collections import namedtuple, defaultdict
from decimal import Decimal

import numpy as np
from scipy.special import comb

from glycopeptidepy.utils.memoize import memoize
from glycopeptidepy.algorithm import PeptidoformGenerator

from ms_deisotope.peak_set import window_peak_set


@memoize(100000000000)
def binomial_pmf(n, i, p):
    try:
        return comb(n, i, exact=True) * (p ** i) * ((1 - p) ** (n - i))
    except OverflowError:
        dn = Decimal(n)
        di = Decimal(i)
        dp = Decimal(p)
        x = math.factorial(dn) / (math.factorial(di) * math.factorial(dn - di))
        return float(x * dp ** di * ((1 - dp) ** (dn - di)))


class PeakWindow(object):
    def __init__(self, peaks):
        self.peaks = list(peaks)
        self.max_mass = 0
        self._calculate()

    def __iter__(self):
        return iter(self.peaks)

    def __getitem__(self, i):
        return self.peaks[i]

    def __len__(self):
        return len(self.peaks)

    def _calculate(self):
        self.peaks.sort(key=lambda x: x.intensity, reverse=True)
        self.max_mass = 0
        for peak in self.peaks:
            if peak.neutral_mass > self.max_mass:
                self.max_mass = peak.neutral_mass

    def __repr__(self):
        template = "{self.__class__.__name__}({self.max_mass}, {size})"
        return template.format(self=self, size=len(self))


AScoreSolution = namedtuple("AScoreSolution", ["target", "a_score", "sites"])
AScoreCandidate = namedtuple("AScoreCandidate", ["peptide", "modifications"])


class AScoreEvaluator(object):
    def __init__(self, scan, peptide, modification_rule, n_positions=1):
        self._scan = None
        self.peak_windows = None
        self.scan = scan
        self.peptide = peptide
        self.modification_rule = modification_rule
        self.n_positions = n_positions
        self.peptidoforms = self.generate_peptidoforms(self.modification_rule)
        self._fragment_cache = {}

    @property
    def scan(self):
        return self._scan

    @scan.setter
    def scan(self, value):
        self._scan = value
        if value is None:
            self.peak_windows = []
        else:
            self.peak_windows = map(PeakWindow, window_peak_set(value.deconvoluted_peak_set))

    def find_existing(self, modification_rule):
        indices = []
        for i, position in enumerate(self.peptide):
            if modification_rule in position.modifications:
                indices.append(i)
        return indices

    def generate_base_peptides(self, modification_rule):
        existing_indices = self.find_existing(modification_rule)
        base_peptides = []
        for indices in itertools.combinations(existing_indices, self.n_positions):
            base_peptide = self.peptide.clone()
            for i in indices:
                base_peptide.drop_modification(i, modification_rule)
            base_peptides.append(base_peptide)
        return base_peptides

    def generate_peptidoforms(self, modification_rule):
        base_peptides = self.generate_base_peptides(modification_rule)
        pepgen = PeptidoformGenerator([], [modification_rule], self.n_positions)
        peptidoforms = defaultdict(set)
        for base_peptide in base_peptides:
            mod_combos = pepgen.modification_sites(base_peptide)
            for mod_combo in mod_combos:
                if len(mod_combo) != self.n_positions:
                    continue
                peptidoform, _n_mods = pepgen.apply_variable_modifications(
                    base_peptide, mod_combo, None, None)
                peptidoforms[peptidoform].add(tuple(mod_combo))
        return [AScoreCandidate(peptide, mods) for peptide, mods in  peptidoforms.items()]

    def _generate_fragments(self, peptidoform):
        try:
            return self._fragment_cache[peptidoform]
        except KeyError:
            frags = itertools.chain.from_iterable(
                itertools.chain(
                    peptidoform.get_fragments("y"),
                    peptidoform.get_fragments("b")))
            frags = list(frags)
            frags.sort(key=lambda x: x.mass)
            self._fragment_cache[peptidoform] = frags
            return frags

    def match_ions(self, peptidoform, depth=10, error_tolerance=1e-5):
        frags = self._generate_fragments(peptidoform)
        n = 0
        window_i = 0
        window_n = len(self.peak_windows)
        current_window = self.peak_windows[window_i]
        for frag in frags:
            while not current_window or (frag.mass >= (current_window.max_mass + 1)):
                window_i += 1
                if window_i == window_n:
                    return n
                current_window = self.peak_windows[window_i]
            for peak in current_window[:depth]:
                if abs(peak.neutral_mass - frag.mass) / frag.mass < error_tolerance:
                    n += 1
        return n

    def permutation_score(self, peptidoform):
        frags = self._generate_fragments(peptidoform)
        N = len(frags)
        site_scores = np.zeros(10)
        for i in range(1, 11):
            n = self.match_ions(peptidoform, i)
            p = i / 100.0
            cumulative_score = binomial_pmf(N, n, p)
            site_scores[i - 1] = (abs(-10.0 * math.log10(cumulative_score)))
        return site_scores

    def rank_permutations(self, permutation_scores):
        ranking = []
        for i in range(len(permutation_scores)):
            weighted_score = self._score(permutation_scores[i])
            ranking.append((weighted_score, i))
        ranking.sort(reverse=True)
        return ranking

    _weight_vector = np.array([
        0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, .25, .25
    ])

    def _score(self, scores):
        return self._weight_vector.dot(scores) / 10.0

    def score(self, error_tolerance=1e-5):
        scores = [self.permutation_score(candidate.peptide) for candidate in self.peptidoforms]
        ranked = self.rank_permutations(scores)
        solutions = [AScoreSolution(self.peptidoforms[i].peptide, score, self.peptidoforms[i].modifications) for score, i in ranked]
        return solutions

