from glypy.composition.glycan_composition import FrozenGlycanComposition, FrozenMonosaccharideResidue


_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")


def composition_distance(c1, c2):
    keys = set(c1) | set(c2)
    distance = 0
    for k in keys:
        distance += abs(c1[k] - c2[k])
    return distance, 1.0


def n_glycan_distance(c1, c2):
    distance, weight = composition_distance(c1, c2)
    if abs(c1[_hexose] - c2[_hexose]) == 1 and abs(c1[_hexnac] - c2[_hexnac]) == 1:
        distance -= 1
    else:
        if c1[_hexose] == c1[_hexnac] or c2[_hexose] == c2[_hexnac]:
            weight /= 2.
    return distance, weight


def mass_diff(c1, c2):
    return abs(c1.composition.mass() - c2.composition.mass()) > 600


class CompositionGraphNode(object):
    def __init__(self, composition, index, **kwargs):
        self.composition = composition
        self.index = index
        self.edges = set()
        self._str = str(self.composition)
        self._hash = hash(str(self.composition))

    @property
    def order(self):
        return len(self.edges)

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self._str

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return "CompositionGraphNode(%s, %d)" % ((self.composition), len(self.edges))


class CompositionGraphEdge(object):
    def __init__(self, node1, node2, order, weight=1.0):
        self.node1 = node1
        self.node2 = node2
        self.order = order
        self.weight = weight
        self._hash = hash((node1, node2, order))
        self._str = "(%s)" % ', '.join(map(str, (node1, node2, order)))
        node1.edges.add(self)
        node2.edges.add(self)

    def __getitem__(self, key):
        if key == self.node1:
            return self.node2
        elif key == self.node2:
            return self.node1
        else:
            raise KeyError(key)

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self._str

    def __hash__(self):
        return self._hash


class CompositionGraph(object):
    def __init__(self, compositions):
        self.nodes = []
        self.node_map = {}
        self._create_nodes(compositions)
        self.edges = []

    def _create_nodes(self, compositions):
        i = 0
        if isinstance(compositions[0], basestring):
            compositions = map(FrozenGlycanComposition.parse, compositions)

        for c in compositions:
            n = CompositionGraphNode(c, i)
            self.nodes.append(n)
            self.node_map[str(n.composition)] = n
            i += 1

    def _create_edges(self, degree=1, distance_fn=composition_distance, threshold_fn=mass_diff):
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                c1 = self.nodes[i]
                c2 = self.nodes[j]
                diff, weight = distance_fn(c1.composition, c2.composition)
                if diff <= degree:
                    e = CompositionGraphEdge(c1, c2, diff, weight)
                    self.edges.append(e)
                if threshold_fn(c1, c2):
                    break

    def __getitem__(self, key):
        return self.node_map[key]
