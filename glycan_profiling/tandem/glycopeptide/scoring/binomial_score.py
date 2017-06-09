# -*- coding: utf-8 -*-

'''
Much of this logic is derived from:

    Risk, B. A., Edwards, N. J., & Giddings, M. C. (2013). A peptide-spectrum scoring system
    based on ion alignment, intensity, and pair probabilities. Journal of Proteome Research,
    12(9), 4240–7. http://doi.org/10.1021/pr400286p
'''


import numpy as np
from scipy.misc import comb

from glycopeptidepy.utils.memoize import memoize

from .base import GlycopeptideSpectrumMatcherBase
from .fragment_match_map import FragmentMatchMap


@memoize(100000000000)
def binomial_pmf(n, i, p):
    return comb(n, i, exact=True) * (p ** i) * ((1 - p) ** (n - i))


@memoize(100000000000)
def binomial_tail_probability(n, k, p):
    total = 0.0
    for i in range(k, n):
        v = binomial_pmf(n, i, p)
        if np.isnan(v):
            continue
        total += v
    return total


def binomial_fragments_matched(total_product_ion_count, count_product_ion_matches, ion_tolerance,
                               precursor_mass):
    p = np.exp((np.log(ion_tolerance) + np.log(2)) +
               np.log(count_product_ion_matches) - np.log(precursor_mass))
    return binomial_tail_probability(total_product_ion_count, count_product_ion_matches, p)


def median_sorted(numbers):
    n = len(numbers)
    if n == 0:
        return (n - 1) / 2, 0
    elif n % 2 == 0:
        return (n - 1) / 2, (numbers[(n - 1) / 2] + numbers[((n - 1) / 2) + 1]) / 2.
    else:
        return (n - 1) / 2, numbers[(n - 1) / 2]


def medians(array):
    array.sort()
    offset, m1 = median_sorted(array)
    offset += 1
    i, m2 = median_sorted(array[offset:])
    offset += i + 1
    i, m3 = median_sorted(array[offset:])
    offset += i + 1
    i, m4 = median_sorted(array[offset:])
    return m1, m2, m3, m4


def _counting_tiers(peak_list, matched_peaks, total_product_ion_count):
    intensity_list = np.array([p.intensity for p in peak_list])
    m1, m2, m3, m4 = medians(intensity_list)

    matched_intensities = np.array(
        [p.intensity for match, p in matched_peaks.items()])
    counts = dict()
    next_count = (matched_intensities > m1).sum()
    counts[1] = next_count

    next_count = (matched_intensities > m2).sum()
    counts[2] = next_count

    next_count = (matched_intensities > m3).sum()
    counts[3] = next_count

    next_count = (matched_intensities > m4).sum()
    counts[4] = next_count
    return counts


def _intensity_tiers(peak_list, matched_peaks, total_product_ion_count):
    intensity_list = np.array([p.intensity for p in peak_list])
    m1, m2, m3, m4 = medians(intensity_list)

    matched_intensities = np.array(
        [p.intensity for match, p in matched_peaks.items()])
    counts = dict()
    last_count = total_product_ion_count
    next_count = (matched_intensities > m1).sum()
    counts[1] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m2).sum()
    counts[2] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m3).sum()
    counts[3] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m4).sum()
    counts[4] = binomial_tail_probability(last_count, next_count, 0.5)
    return counts


def _score_tiers(peak_list, matched_peaks, total_product_ion_count):
    intensity_list = np.array([p.score for p in peak_list])
    m1, m2, m3, m4 = medians(intensity_list)

    matched_intensities = np.array(
        [p.score for match, p in matched_peaks.items()])
    counts = dict()
    last_count = total_product_ion_count
    next_count = (matched_intensities > m1).sum()
    counts[1] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m2).sum()
    counts[2] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m3).sum()
    counts[3] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m4).sum()
    counts[4] = binomial_tail_probability(last_count, next_count, 0.5)
    return counts


def binomial_intensity(peak_list, matched_peaks, total_product_ion_count):
    if len(matched_peaks) == 0:
        return np.exp(0)
    counts = _intensity_tiers(peak_list, matched_peaks, total_product_ion_count)

    prod = 0
    for v in counts.values():
        if v == 0:
            continue
        prod += np.log(v)
    return np.exp(prod)


def calculate_precursor_mass(spectrum_match):
    precursor_mass = spectrum_match.target.total_composition().mass
    return precursor_mass


class BinomialSpectrumMatcher(GlycopeptideSpectrumMatcherBase):

    def __init__(self, scan, target):
        super(BinomialSpectrumMatcher, self).__init__(scan, target)
        self._sanitized_spectrum = set(self.spectrum)
        self._score = None
        self.solution_map = FragmentMatchMap()
        self.n_theoretical = 0
        self._backbone_mass_series = []

    def match(self, error_tolerance=2e-5):
        n_theoretical = 0
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
        backbone_mass_series = []

        oxonium_ion_matches = set()
        for frag in self.target.glycan_fragments(
                all_series=False, allow_ambiguous=False,
                include_large_glycan_fragments=False,
                maximum_fragment_size=4):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                solution_map.add(peak, frag)
                oxonium_ion_matches.add(peak)
                try:
                    self._sanitized_spectrum.remove(peak)
                except KeyError:
                    continue
        for frags in self.target.get_fragments('b'):
            for frag in frags:
                backbone_mass_series.append(frag.mass)
                n_theoretical += 1
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak in oxonium_ion_matches:
                        continue
                    solution_map.add(peak, frag)
                self._backbone_mass_series
        for frags in self.target.get_fragments('y'):
            backbone_mass_series.append(frag.mass)
            for frag in frags:
                n_theoretical += 1
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak in oxonium_ion_matches:
                        continue
                    solution_map.add(peak, frag)
        for frag in self.target.stub_fragments(extended=True):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)
        self.solution_map = solution_map
        self.n_theoretical = n_theoretical
        self._backbone_mass_series = backbone_mass_series
        return solution_map

    def _sanitize_solution_map(self):
        san = FragmentMatchMap()
        for pair in self.solution_map:
            if pair.fragment.series != "oxonium_ion":
                san.add(pair)
        return san

    def _compute_average_window_size(self, match_tolerance=2e-5):
        # window_sizes = [
        #     match_tolerance * frag.mass * 2
        #     for frag in self._backbone_mass_series
        # ]

        # average_window_size = sum(window_sizes) / len(window_sizes)
        average_window_size = (
            (self.target.peptide_composition(
            ).mass) / 3.) * match_tolerance * 2
        return average_window_size

    def _fragment_matched_binomial(self, match_tolerance=2e-5):
        precursor_mass = calculate_precursor_mass(self)

        fragment_match_component = binomial_fragments_matched(
            self.n_theoretical,
            len(self._sanitize_solution_map()),
            self._compute_average_window_size(match_tolerance),
            precursor_mass
        )
        return fragment_match_component

    def _intensity_component_binomial(self):
        intensity_component = binomial_intensity(
            self._sanitized_spectrum,
            self._sanitize_solution_map(),
            self.n_theoretical)

        if intensity_component == 0:
            intensity_component = 1e-170
        return intensity_component

    def _binomial_score(self, match_tolerance=2e-5, *args, **kwargs):
        precursor_mass = calculate_precursor_mass(self)

        solution_map = self._sanitize_solution_map()
        n_matched = len(solution_map)
        if n_matched == 0 or len(self._sanitized_spectrum) == 0:
            return 0

        fragment_match_component = binomial_fragments_matched(
            self.n_theoretical,
            len(self._sanitize_solution_map()),
            self._compute_average_window_size(match_tolerance),
            precursor_mass
        )

        if fragment_match_component < 1e-170:
            fragment_match_component = 1e-170

        intensity_component = binomial_intensity(
            self._sanitized_spectrum,
            solution_map,
            self.n_theoretical)

        if intensity_component == 0:
            intensity_component = 1e-170
        score = -np.log10(intensity_component) + -np.log10(fragment_match_component)

        if np.isinf(score):
            print("infinite score", self.scan, self.target, intensity_component, fragment_match_component, self.scan)

        return score

    def calculate_score(self, match_tolerance=2e-5, *args, **kwargs):
        score = self._binomial_score(match_tolerance)
        self._score = score
        return score
