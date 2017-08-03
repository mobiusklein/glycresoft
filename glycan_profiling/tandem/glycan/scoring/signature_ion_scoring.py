import itertools

from collections import Counter

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

    def match(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        glycan_composition = self.target
        peak_set = self.spectrum
        matches = FragmentMatchMap()
        max_peak = max([p.intensity for p in peak_set])

        # Simple oxonium ions
        for k in glycan_composition.keys():
            # Fucose does not produce a reliable oxonium ion
            if is_fucose(k):
                continue
            f = Fragment('B', {}, [], k.mass(), name=str(k),
                         composition=k.total_composition())
            for hit in peak_set.all_peaks_for(f.mass, error_tolerance):
                if hit.intensity / max_peak < 0.01:
                    continue
                matches.add(hit, f)

        # Compound oxonium ions
        if include_compound:
            for i in range(2, 4):
                for kk in itertools.combinations_with_replacement(sorted(glycan_composition, key=str), i):
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
        self.solution_map = matches
        return matches

    def calculate_score(self, error_tolerance=2e-5, include_compound=False, *args, **kwargs):
        imax = max(self.spectrum, key=lambda x: x.intensity).intensity
        oxonium = 0
        n = 0
        for peak, fragment in self.solution_map:
            oxonium += peak.intensity / imax
            n += 1
        if n == 0:
            return 0
        self._score = oxonium / n
        return self._score
