from glypy.structure.glycan_composition import FrozenMonosaccharideResidue

from glycan_profiling.database.glycan_composition_filter import GlycanCompositionFilter

_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")


def composition_distance(c1, c2):
    '''N-Dimensional Manhattan Distance or L1 Norm
    '''
    keys = set(c1) | set(c2)
    distance = 0.0
    for k in keys:
        distance += abs(c1[k] - c2[k])
    return int(distance), 1 / distance if distance > 0 else 1


def n_glycan_distance(c1, c2):
    distance, weight = composition_distance(c1, c2)
    # if abs(c1[_hexose] - c2[_hexose]) == 1 and abs(c1[_hexnac] - c2[_hexnac]) == 1:
    #     distance -= 1
    # else:
    #     if c1[_hexose] == c1[_hexnac] or c2[_hexose] == c2[_hexnac]:
    #         weight /= 2.
    return distance, weight


class CompositionSpace(object):

    def __init__(self, members):
        self.filter = GlycanCompositionFilter(members)

    @property
    def monosaccharides(self):
        return self.filter.monosaccharides

    def find_narrowly_related(self, composition, window=1):
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
