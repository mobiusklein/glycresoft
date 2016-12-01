import numpy as np

try:
    from StringIO import StringIO
except:
    from io import StringIO

try:
    basestring
except:
    basestring = (str, bytes)

from glypy.composition.glycan_composition import FrozenMonosaccharideResidue, GlycanComposition
from glycopeptidepy import HashableGlycanComposition
from glycopeptidepy.structure.glycan import GlycanCompositionProxy


from .glycan_composition_filter import GlycanCompositionFilter


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

    def find_related(self, composition, window=1):
        m = self.monosaccharides[0]
        q = self.filter.query(
            m, composition[m] - window, composition[m] + window)
        for m in self.monosaccharides[1:]:
            center = composition[m]
            q.add(m, center - window, center + window)
        return q.all()


class CompositionGraphNode(object):
    _temp_score = 0.0

    def __init__(self, composition, index, score=0., **kwargs):
        self.composition = composition
        self.index = index
        self.edges = set()
        self._str = str(self.composition)
        self._hash = hash(str(self._str))
        self.score = score
        self.internal_score = 0.0

    @property
    def glycan_composition(self):
        return self.composition

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
        return "CompositionGraphNode(%s, %d, %0.2e)" % ((self._str), len(self.edges), self.score)

    def clone(self):
        return CompositionGraphNode(self.composition, self.index, self.score)


class CompositionGraphEdge(object):
    __slots__ = ["node1", "node2", "order", "weight", "_hash", "_str"]

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

    def __repr__(self):
        return self._str

    def __hash__(self):
        return self._hash

    def __reduce__(self):
        return self.__class__, (self.node1, self.node2, self.order, self.weight)

    def copy_for(self, node1, node2):
        return self.__class__(node1, node2, self.order, self.weight)


class CompositionGraph(object):

    def __init__(self, compositions):
        self.nodes = []
        self.node_map = {}
        self.create_nodes(compositions)
        self.edges = []

    def create_nodes(self, compositions):
        i = 0
        compositions = map(HashableGlycanComposition.parse, compositions)

        for c in compositions:
            n = CompositionGraphNode(c, i)
            self.add_node(n)
            i += 1

    def add_node(self, node):
        self.nodes.append(node)
        self.node_map[str(node.composition)] = node

    def create_edges(self, degree=2, distance_fn=composition_distance):
        space = CompositionSpace([node.composition for node in self])
        for node in self:
            for related in space.find_narrowly_related(node.composition, degree):
                related_node = self.node_map[str(related)]
                # Ensures no duplicates assuming symmetric search window
                if node.index < related_node.index:
                    diff, weight = distance_fn(node.composition, related)
                    e = CompositionGraphEdge(node, related_node, diff, weight)
                    self.edges.append(e)

    def __getitem__(self, key):
        if isinstance(key, (basestring, GlycanComposition, GlycanCompositionProxy)):
            return self.node_map[key]
        elif isinstance(key, int):
            return self.nodes[key]
        else:
            try:
                result = []
                iter(key)
                if isinstance(key[0], (bool, np.bool_)):
                    for b, k in zip(key, self):
                        if b:
                            result.append(k)
                else:
                    for k in key:
                        result.append(self[k])
                return result
            except Exception, e:
                raise IndexError(
                    "An error occurred (%r) during indexing with %r" % (e, key))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getstate__(self):
        string_buffer = StringIO()
        dump(self, string_buffer)
        return string_buffer

    def __setstate__(self, string_buffer):
        string_buffer.seek(0)
        net = load(string_buffer)
        self.nodes = net.nodes
        self.node_map = net.node_map
        self.edges = net.edges

    def clone(self):
        graph = CompositionGraph([])
        for node in self.nodes:
            graph.add_node(node.clone())
        for edge in self.edges:
            n1 = graph.nodes[edge.node1.index]
            n2 = graph.nodes[edge.node2.index]
            e = edge.copy_for(n1, n2)
            graph.edges.append(e)
        return graph


class GraphWriter(object):

    def __init__(self, network, file_obj):
        self.network = network
        self.file_obj = file_obj
        self.handle_graph(self.network)

    def handle_node(self, node):
        composition = node.composition.serialize()
        index = (node.index)
        score = node.score
        self.file_obj.write("%d\t%s\t%f\n" % (index, composition, score))

    def handle_edge(self, edge):
        index1 = edge.node1.index
        index2 = edge.node2.index
        diff = edge.order
        weight = edge.weight
        self.file_obj.write("%d\t%d\t%d\t%f\n" %
                            (index1, index2, diff, weight))

    def handle_graph(self, graph):
        self.file_obj.write("#COMPOSITIONGRAPH 1.0\n")
        self.file_obj.write("#NODE\n")
        for node in self.network.nodes:
            self.handle_node(node)
        self.file_obj.write("#EDGE\n")
        for edge in self.network.edges:
            self.handle_edge(edge)


class GraphReader(object):

    @classmethod
    def read(cls, file_obj):
        return cls(file_obj).network

    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.network = CompositionGraph([])
        self.handle_graph_file()

    def handle_node_line(self, line):
        try:
            index, composition, score = line.replace("\n", "").split("\t")
            score = float(score)
        except ValueError:
            index, composition = line.replace("\n", "").split("\t")
            score = 0.

        index = int(index)
        composition = HashableGlycanComposition.parse(composition)
        node = CompositionGraphNode(composition, index, score)
        self.network.nodes.append(node)
        self.network.node_map[str(node.composition)] = node

    def handle_edge_line(self, line):
        index1, index2, diff, weight = line.replace("\n", "").split("\t")
        index1, index2, diff, weight = int(index1), int(
            index2), int(diff), float(weight)
        node1 = self.network.nodes[index1]
        node2 = self.network.nodes[index2]
        edge = CompositionGraphEdge(node1, node2, diff, weight)
        self.network.edges.append(edge)

    def handle_graph_file(self):
        state = "START"
        for line in self.file_obj:
            line = line.strip()
            if state == "START":
                if line == "#NODE":
                    state = "NODE"
            elif state == "NODE":
                if line == "#EDGE":
                    state = "EDGE"
                else:
                    self.handle_node_line(line)
            elif state == "EDGE":
                self.handle_edge_line(line)


dump = GraphWriter
load = GraphReader.read
