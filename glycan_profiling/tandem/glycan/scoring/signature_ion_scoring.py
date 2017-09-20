import itertools

from collections import Counter

import numpy as np

from glypy.structure.fragment import Fragment
from glypy.composition import Composition
from glypy.composition.composition_transform import strip_derivatization
from glypy.composition.glycan_composition import MonosaccharideResidue
from glypy.io.nomenclature.identity import is_a

from glycan_profiling.structure import FragmentMatchMap
from glycan_profiling.structure.fragment_match_map import SpectrumGraph
from glycan_profiling.tandem.spectrum_matcher_base import SpectrumMatcherBase

from glycopeptidepy.utils.memoize import memoize

fucose = MonosaccharideResidue.from_iupac_lite("Fuc")


@memoize(100000000000)
def is_fucose(residue):
    return is_a(
        strip_derivatization(residue.clone(
            monosaccharide_type=MonosaccharideResidue)), fucose)


class SignatureIonScorer(SpectrumMatcherBase):
    def __init__(self, scan, glycan_composition):
        super(SignatureIonScorer, self).__init__(scan, glycan_composition)
        self.fragments_searched = 0
        self.fragments_matched = 0
        self.minimum_intensity_threshold = 0.01

    def _get_matched_peaks(self):
        peaks = set()
        for peak in self.solution_map.by_peak:
            peaks.add(peak)
        for p1, p2, label in self.spectrum_graph:
            peaks.add(p1)
            peaks.add(p2)
        return peaks

    def percent_matched_intensity(self):
        matched = 0
        total = 0

        for peak in self._get_matched_peaks():
            matched += peak.intensity

        for peak in self.spectrum:
            total += peak.intensity
        return matched / total

    def _find_peak_pairs(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        peak_set = self.spectrum
        pairs = SpectrumGraph()

        blocks = [(part, part.mass()) for part in self.target if not is_fucose(part)]
        if include_compound:
            compound_blocks = list(itertools.combinations(self.target, 2))
            compound_blocks = [(block, sum(part.mass() for part in block))
                               for block in compound_blocks]
            blocks.extend(compound_blocks)
        try:
            max_peak = max([p.intensity for p in peak_set])
            threshold = max_peak * self.minimum_intensity_threshold
        except ValueError:
            return []

        for peak in peak_set:
            if peak.intensity < threshold or peak.neutral_mass < 150:
                continue
            for block, mass in blocks:
                for other in peak_set.all_peaks_for(peak.neutral_mass + mass, error_tolerance):
                    if other.intensity < threshold:
                        continue
                    pairs.add(peak, other, block)
        return pairs

    def match(self, error_tolerance=2e-5, include_compound=False, combination_size=3, *args, **kwargs):
        glycan_composition = self.target
        peak_set = self.spectrum
        matches = FragmentMatchMap()
        water = Composition("H2O")
        counter = 0
        try:
            max_peak = max([p.intensity for p in peak_set])
            threshold = max_peak * self.minimum_intensity_threshold
        except ValueError:
            self.solution_map = matches
            self.fragments_searched = counter
            self.pairs = SpectrumGraph()
            return matches
        # Simple oxonium ions
        for k in glycan_composition.keys():
            # Fucose does not produce a reliable oxonium ion
            if is_fucose(k):
                continue
            counter += 1
            f = Fragment('B', {}, [], k.mass(), name=str(k),
                         composition=k.total_composition())
            for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                if hit.intensity < threshold:
                    continue
                matches.add(hit, f)
            f = Fragment('B', {}, [], k.mass() - water.mass, name="%s-H2O" % str(k),
                         composition=k.total_composition() - water)
            for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                if hit.intensity / max_peak < self.minimum_intensity_threshold:
                    continue
                matches.add(hit, f)

        # Compound oxonium ions
        if include_compound:
            for i in range(2, combination_size + 1):
                for kk in itertools.combinations_with_replacement(sorted(glycan_composition, key=str), i):
                    counter += 1
                    invalid = False
                    for k, v in Counter(kk).items():
                        if glycan_composition[k] < v:
                            invalid = True
                            break
                    if invalid:
                        continue
                    key = '-'.join(map(str, kk))
                    mass = sum(k.mass() for k in kk)
                    composition = sum((k.total_composition() for k in kk), Composition())
                    f = Fragment('B', {}, [], mass, name=key,
                                 composition=composition)
                    for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                        if hit.intensity / max_peak < 0.01:
                            continue
                        matches.add(hit, f)

                    f = Fragment('B', {}, [], mass - water.mass, name="%s-H2O" % key,
                                 composition=composition - water)
                    for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                        if hit.intensity / max_peak < 0.01:
                            continue
                        matches.add(hit, f)

        self.spectrum_graph = self._find_peak_pairs(error_tolerance, include_compound)
        self.solution_map = matches
        self.fragments_searched = counter
        return matches

    def penalize(self, ratio, count):
        count = float(count)
        scale = min(np.log(count) / np.log(4), 1)
        return ratio * scale

    def oxonium_ratio(self):
        imax = max(self.spectrum, key=lambda x: x.intensity).intensity
        oxonium = 0
        n = 0
        for peak, fragment in self.solution_map:
            oxonium += peak.intensity / imax
            n += 1
        self.fragments_matched = n
        if n == 0:
            return 0, 0
        return (oxonium / n), n

    def _score_pairs(self):
        try:
            imax = max(self.spectrum, key=lambda x: x.intensity).intensity
        except ValueError:
            return 0
        edge_weight = 0
        n = 0
        for p1, p2, label in self.spectrum_graph:
            edge_weight += (p1.intensity + p2.intensity) / imax
            n += 1
        if n == 0:
            return 0
        scale = max(min(np.log(n) / np.log(4), 1), 0.01)
        return (edge_weight / n) * scale

    def calculate_score(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        if len(self.spectrum) == 0:
            self._score = 0
            return self._score
        try:
            oxonium_ratio, n = self.oxonium_ratio()
        except ValueError:
            oxonium_ratio, n = 0, 0
        if n == 0:
            self._score = 0
        else:
            # simple glycan compositions like high mannose N-glycans don't produce
            # many distinct oxonium ions
            if len(self.target.items()) > 2:
                self._score = self.penalize(oxonium_ratio, n)
            else:
                self._score = oxonium_ratio
        edge_weight = self._score_pairs()
        if edge_weight > self._score:
            self._score = edge_weight
        return self._score
