# -*- coding: utf-8 -*-

'''
Much of this logic is derived from:

    Risk, B. A., Edwards, N. J., & Giddings, M. C. (2013). A peptide-spectrum scoring system
    based on ion alignment, intensity, and pair probabilities. Journal of Proteome Research,
    12(9), 4240–7. http://doi.org/10.1021/pr400286p
'''


import numpy as np
from scipy.special import comb
from decimal import Decimal
import math

from glycopeptidepy.utils.memoize import memoize

from .base import (
    GlycopeptideSpectrumMatcherBase, ChemicalShift, EXDFragmentationStrategy,
    HCDFragmentationStrategy, IonSeries)
from glycresoft.structure import FragmentMatchMap


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
        return (n - 1) // 2, 0
    elif n % 2 == 0:
        return (n - 1) // 2, (numbers[(n - 1) // 2] + numbers[((n - 1) // 2) + 1]) / 2.
    else:
        return (n - 1) // 2, numbers[(n - 1) // 2]


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
        [p.intensity for p, _ in matched_peaks])
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
        [p.intensity for p, _ in matched_peaks])
    counts = dict()
    last_count = total_product_ion_count
    next_count = (matched_intensities > m1).sum()
    if last_count == 0:
        counts[1] = 1.0
    elif last_count == next_count:
        counts[1] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[1] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m2).sum()
    if last_count == 0:
        counts[2] = 1.0
    elif last_count == next_count:
        counts[2] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[2] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m3).sum()
    if last_count == 0:
        counts[3] = 1.0
    elif last_count == next_count:
        counts[3] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[3] = binomial_tail_probability(last_count, next_count, 0.5)

    last_count = next_count

    next_count = (matched_intensities > m4).sum()
    if last_count == 0:
        counts[4] = 1.0
    elif last_count == next_count:
        counts[4] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
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
    if last_count == 0:
        counts[1] = 1.0
    elif last_count == next_count:
        counts[1] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[1] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m2).sum()
    if last_count == 0:
        counts[2] = 1.0
    elif last_count == next_count:
        counts[2] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[2] = binomial_tail_probability(last_count, next_count, 0.5)
    last_count = next_count

    next_count = (matched_intensities > m3).sum()
    if last_count == 0:
        counts[3] = 1.0
    elif last_count == next_count:
        counts[3] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[3] = binomial_tail_probability(last_count, next_count, 0.5)

    last_count = next_count

    next_count = (matched_intensities > m4).sum()
    if last_count == 0:
        counts[4] = 1.0
    elif last_count == next_count:
        counts[4] = binomial_tail_probability(last_count, next_count - 1, 0.5)
    else:
        counts[4] = binomial_tail_probability(last_count, next_count, 0.5)
    return counts


def binomial_intensity(peak_list, matched_peaks, total_product_ion_count):
    if len(matched_peaks) == 0:
        return np.exp(0)
    counts = _intensity_tiers(peak_list, matched_peaks, total_product_ion_count)

    prod = 0
    for k, v in counts.items():
        if v == 0:
            v = 1e-20
        prod += np.log(v)
    return np.exp(prod)


def calculate_precursor_mass(spectrum_match):
    precursor_mass = spectrum_match.target.total_composition().mass
    return precursor_mass


class BinomialSpectrumMatcher(GlycopeptideSpectrumMatcherBase):

    def __init__(self, scan, target, mass_shift=None):
        super(BinomialSpectrumMatcher, self).__init__(scan, target, mass_shift)
        self.solution_map = FragmentMatchMap()
        self._init_binomial()

    def _init_binomial(self):
        self._sanitized_spectrum = set(self.spectrum)
        self.n_theoretical = 0

    def _match_oxonium_ions(self, error_tolerance=2e-5, masked_peaks=None):
        if masked_peaks is None:
            masked_peaks = set()
        val = super(BinomialSpectrumMatcher, self)._match_oxonium_ions(
            error_tolerance=error_tolerance, masked_peaks=masked_peaks)
        self._sanitized_spectrum -= {self.spectrum[i] for i in masked_peaks}
        return val

    def _match_backbone_series(self, series, error_tolerance=2e-5, masked_peaks=None, strategy=None,
                               include_neutral_losses=False):
        if strategy is None:
            strategy = HCDFragmentationStrategy
        for frags in self.get_fragments(series, strategy=strategy, include_neutral_losses=include_neutral_losses):
            # Should this be on the level of position, or the level of the individual fragment ions?
            # At the level of position, this makes missing only glycosylated or unglycosylated ions
            # less punishing, while at the level of the fragment makes more sense by the definition
            # of the geometric mass accuracy interpretation.
            #
            # Using the less severe case to be less pessimistic
            self.n_theoretical += 1
            for frag in frags:
                for peak in self.spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    self.solution_map.add(peak, frag)

    def _sanitize_solution_map(self):
        san = list()
        for pair in self.solution_map:
            if pair.fragment.series != "oxonium_ion":
                san.append(pair)
        return san

    def _compute_average_window_size(self, error_tolerance=2e-5):
        average_window_size = (
            (self.target.peptide_composition(
            ).mass) / 3.) * error_tolerance * 2
        return average_window_size

    def _fragment_matched_binomial(self, error_tolerance=2e-5):
        precursor_mass = calculate_precursor_mass(self)

        fragment_match_component = binomial_fragments_matched(
            self.n_theoretical,
            len(self._sanitize_solution_map()),
            self._compute_average_window_size(error_tolerance),
            precursor_mass
        )
        if fragment_match_component < 1e-170:
            fragment_match_component = 1e-170
        return fragment_match_component

    def _intensity_component_binomial(self):
        intensity_component = binomial_intensity(
            self._sanitized_spectrum,
            self._sanitize_solution_map(),
            self.n_theoretical)

        if intensity_component < 1e-170:
            intensity_component = 1e-170
        return intensity_component

    def _binomial_score(self, error_tolerance=2e-5, *args, **kwargs):
        precursor_mass = calculate_precursor_mass(self)

        solution_map = self._sanitize_solution_map()
        n_matched = len(solution_map)
        if n_matched == 0 or len(self._sanitized_spectrum) == 0:
            return 0

        fragment_match_component = binomial_fragments_matched(
            self.n_theoretical,
            len(solution_map),
            self._compute_average_window_size(error_tolerance),
            precursor_mass
        )

        if fragment_match_component < 1e-170:
            fragment_match_component = 1e-170

        intensity_component = binomial_intensity(
            self._sanitized_spectrum,
            solution_map,
            self.n_theoretical)

        if intensity_component < 1e-170:
            intensity_component = 1e-170
        score = -np.log10(intensity_component) + -np.log10(fragment_match_component)

        if np.isinf(score):
            print("infinite score", self.scan, self.target, intensity_component, fragment_match_component, self.scan)

        return score

    def calculate_score(self, error_tolerance=2e-5, *args, **kwargs):
        score = self._binomial_score(error_tolerance)
        self._score = score
        return score


class StubIgnoringBinomialSpectrumMatcher(BinomialSpectrumMatcher):

    def _sanitize_solution_map(self):
        san = list()
        for pair in self.solution_map:
            if pair.fragment.series not in ("oxonium_ion", "stub_glycopeptide"):
                san.append(pair)
        return san
