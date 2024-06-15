from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, HashableGlycanComposition

from glycresoft.database.glycan_composition_filter import GlycanCompositionFilter

from typing import List, Tuple

_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")


def composition_distance(c1: HashableGlycanComposition, c2: HashableGlycanComposition) -> Tuple[int, float]:
    """N-Dimensional Manhattan Distance or L1 Norm"""
    keys = set(c1) | set(c2)
    distance = 0.0
    try:
        c1_get = c1._getitem_fast
        c2_get = c2._getitem_fast
    except AttributeError:
        c1_get = c1.__getitem__
        c2_get = c2.__getitem__
    for k in keys:
        distance += abs(c1_get(k) - c2_get(k))
    return int(distance), 1 / distance if distance > 0 else 1


# Old alias
n_glycan_distance = composition_distance


class CompositionSpace(object):
    filter: GlycanCompositionFilter

    def __init__(self, members):
        self.filter = GlycanCompositionFilter(members)

    @property
    def monosaccharides(self):
        return self.filter.monosaccharides

    def find_narrowly_related(self, composition: HashableGlycanComposition, window: int=1) -> List:
        partitions = []
        for i in range(len(self.monosaccharides)):
            j = 0
            m = self.monosaccharides[j]
            if i == j:
                q = self.filter.query(
                    m, composition[m] - window, composition[m] + window)
            else:
                q = self.filter.query(
                    m, composition[m], composition[m])
            for m in self.monosaccharides[1:]:
                j += 1
                center = composition[m]
                if j == i:
                    q.add(m, center - window, center + window)
                else:
                    q.add(m, center, center)
            partitions.append(q)
        out = set()
        for case in partitions:
            out.update(case)
        return out

    def l1_distance(self, c1, c2):
        keys = set(c1) | set(c2)
        distance = 0
        for k in keys:
            distance += abs(c1[k] - c2[k])
        return distance

    def find_related_broad(self, composition, window=1):
        m = self.monosaccharides[0]
        q = self.filter.query(
            m, composition[m] - window, composition[m] + window)
        for m in self.monosaccharides[1:]:
            center = composition[m]
            q.add(m, center - window, center + window)
        return q.all()

    def find_related(self, composition, window=1):
        if window == 1:
            return self.find_narrowly_related(composition, window)
        candidates = self.find_related_broad(composition, window)
        out = []
        for case in candidates:
            if self.l1_distance(composition, case) <= window:
                out.append(case)
        return out


class DistanceCache(object):
    '''A caching wrapper around a distance function (e.g. composition_distance).
    '''
    def __init__(self, distance_function):
        self.distance_function = distance_function
        self.cache = dict()

    def __call__(self, x, y):
        key = frozenset((x, y))
        try:
            return self.cache[key]
        except KeyError:
            d = self.distance_function(x, y)
            self.cache[key] = d
            return d
