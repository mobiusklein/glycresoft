from scipy.misc import comb
import numpy as np

import math
from collections import defaultdict

from glypy.utils import make_struct


#: Store whether a fragment was observed for the reducing or non-reducing end of a particular glycosidic bond
GlycosidicBondMatch = make_struct("GlycosidicBondMatch", ["reducing", "non_reducing"])

#: Store how many times a fragment was observed for the reducing, non-reducing, or non-bisecting cross-ring
#: permutations of a particular monosaccharide
CrossringMatch = make_struct("CrossringMatch", ["reducing", "non_reducing", "non_bisecting"])


def is_primary_fragment(f):
    return f.break_count == 1


class FragmentScorer(object):
    """Score the data with respect to structure characterization.

    Attributes
    ----------
    crossring_map : dict
        Mapping of residue id to CrossringMatch
    crossring_score : float
        Score derived from cross-ring coverage
    final_score : float
        The final product score combining glycosidic and cross-ring scores
    fragments : list
        The list of matched fragments
    glycosidic_map : dict
        Mapping of link id to GlycosidicBondMatch
    glycosidic_score : float
        Score derived from glycosidic fragments
    structure : StructureRecord
        Source structure to be matched
    """
    def __init__(self, structure, matched_fragments):
        self.structure = structure
        self.fragments = list(filter(is_primary_fragment, matched_fragments))

        self.glycosidic_map = defaultdict(lambda: GlycosidicBondMatch(0, 0))
        self.crossring_map = defaultdict(lambda: CrossringMatch(0, 0, 0))

        self.glycosidic_score = 0.
        self.crossring_score = 0.

        self._score_glycosidic()
        self._score_crossring()

        self.final_score = 0.
        self._score_final()

    def _score_final(self):
        self.final_score = -100 * np.log10((1 - self.glycosidic_score) * (1 - self.crossring_score))

    def _score_glycosidic(self):
        for f in [f for f in self.fragments if not f.crossring_cleavages]:
            link_ind = f.link_ids.keys()[0]
            link_rec = self.glycosidic_map[link_ind]
            if f.kind in "BC":
                link_rec.non_reducing = 1
            else:
                link_rec.reducing = 1

        self.glycosidic_score = sum(map(sum, self.glycosidic_map.values())) / (2. * (len(self.structure) - 1))

    def _score_crossring(self):

        def has_parent(fragment, residue):
            parents = {n.id for p, n in residue.parents()}
            return bool(parents & fragment.included_nodes)

        def has_child(fragment, residue):
            children = {n.id for p, n in residue.children()}
            return bool(children & fragment.included_nodes)

        for f in [f for f in self.fragments if not f.link_ids]:
            residue = self.structure.get(f.crossring_cleavages.keys()[0])
            residue_record = self.crossring_map[residue.id]

            parent = has_parent(f, residue)
            child = has_child(f, residue)

            if parent and child:
                residue_record.non_bisecting += 1
            elif parent:
                residue_record.reducing += 1
            elif child:
                residue_record.non_reducing += 1

        self.crossring_score = sum(map(sum, self.crossring_map.values())) / (3. * len(self.structure))


def binomial_tail_probability(n, k, p):
    total = 0.0
    for i in range(k, n):
        total += comb(n, i, exact=True) * (p ** i) * ((1 - p) ** (n - i))
    return total


def drop_below_intensity(peaks, threshold):
    return [p for p in peaks if p.intensity > threshold]


def median_intensity(peaks):
    return np.median([p.intensity for p in peaks])


SimplePeak = make_struct("SimplePeak", ["intensity"])


epsilon = 1e-10


class IntensityRankScorer(object):
    """Scores an assigned peak list against a uniform intensity
    random model.

    Attributes
    ----------
    assigned_peaks : list
        The collection of assigned and deconvoluted peaks
    final_score : float
        The final score for this data
    medians : dict
        A cache for already computed median intensities
    peaklist : list
        The set of all experimental peaks
    scores : dict
        Scores for each tier
    transformed_peaks : list
        The collection of assigned peaks' most abundant peaks
    """
    def __init__(self, peaklist, assigned_peaks):
        self.peaklist = peaklist
        self.assigned_peaks = assigned_peaks
        self.transformed_peaks = [SimplePeak(
            max([p.intensity for p in sol.fit.experimental])) for sol in assigned_peaks]
        self.scores = {}
        self.medians = {0: 0}
        self.final_score = 0.
        self.rank()

    def rank(self):
        for a in range(1, 5):
            med = median_intensity(
                drop_below_intensity(
                    self.transformed_peaks,
                    self.medians[a - 1]))
            self.medians[a] = med
            i = len(drop_below_intensity(self.transformed_peaks, med))
            n = len(drop_below_intensity(self.peaklist, self.medians[a - 1]))
            p = binomial_tail_probability(n, i, 0.5)
            self.scores[a] = p
        total = 0.
        for p in self.scores.values():
            total += math.log(p + epsilon)
        self.final_score = -np.log10(math.exp(total))
