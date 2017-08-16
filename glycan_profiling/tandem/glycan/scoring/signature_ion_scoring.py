import itertools

from collections import Counter

import numpy as np

from glypy.structure.fragment import Fragment
from glypy.composition import Composition
from glypy.composition.composition_transform import strip_derivatization
from glypy.composition.glycan_composition import MonosaccharideResidue
from glypy.io.nomenclature.identity import is_a

from glycan_profiling.tandem.glycopeptide.scoring.fragment_match_map import FragmentMatchMap
from glycan_profiling.tandem.spectrum_matcher_base import SpectrumMatcherBase

fucose = MonosaccharideResidue.from_iupac_lite("Fuc")


def is_fucose(residue):
    return is_a(
        strip_derivatization(residue.clone(
            monosaccharide_type=MonosaccharideResidue)), fucose)


class SignatureIonScorer(SpectrumMatcherBase):
    def __init__(self, scan, glycan_composition):
        super(SignatureIonScorer, self).__init__(scan, glycan_composition)
        self.fragments_searched = 0
        self.fragments_matched = 0

    def match(self, error_tolerance=2e-5, include_compound=False, combination_size=3, *args, **kwargs):
        glycan_composition = self.target
        peak_set = self.spectrum
        matches = FragmentMatchMap()
        max_peak = max([p.intensity for p in peak_set])
        water = Composition("H2O")
        counter = 0

        # Simple oxonium ions
        for k in glycan_composition.keys():
            # Fucose does not produce a reliable oxonium ion
            if is_fucose(k):
                continue
            counter += 1
            f = Fragment('B', {}, [], k.mass(), name=str(k),
                         composition=k.total_composition())
            for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                if hit.intensity / max_peak < 0.01:
                    continue
                matches.add(hit, f)
            f = Fragment('B', {}, [], k.mass() - water.mass, name="%s-H2O" % str(k),
                         composition=k.total_composition() - water)
            for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                if hit.intensity / max_peak < 0.01:
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
                    key = ''.join(map(str, kk))
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

    def calculate_score(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        oxonium_ratio, n = self.oxonium_ratio()
        if n == 0:
            self._score = 0
        else:
            self._score = self.penalize(oxonium_ratio, n)
        return self._score
