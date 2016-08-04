import numpy as np

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
    _temp_score = 0.0

    def __init__(self, composition, index, score=0., **kwargs):
        self.composition = composition
        self.index = index
        self.edges = set()
        self._str = str(self.composition)
        self._hash = hash(str(self.composition))
        self.score = score
        self.internal_score = 0.0

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
        return "CompositionGraphNode(%s, %d, %0.2f)" % ((self.composition), len(self.edges), self.score)

    def clone(self):
        return CompositionGraphNode(self.composition, self.index, self.score)


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

    def copy_for(self, node1, node2):
        return self.__class__(node1, node2, self.order, self.weight)


class CompositionGraph(object):
    def __init__(self, compositions):
        self.nodes = []
        self.node_map = {}
        if len(compositions) > 0:
            self._create_nodes(compositions)
        self.edges = []

    def _create_nodes(self, compositions):
        i = 0
        if isinstance(compositions[0], basestring):
            compositions = map(FrozenGlycanComposition.parse, compositions)

        for c in compositions:
            n = CompositionGraphNode(c, i)
            self.add_node(n)
            i += 1

    def add_node(self, node):
        self.nodes.append(node)
        self.node_map[str(node.composition)] = node

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
        if isinstance(key, basestring):
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
                raise IndexError("An error occurred (%r) during indexing with %r" % (e, key))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def clone(self):
        graph = CompositionGraph(self, [])
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
        self.file_obj.write("%d\t%d\t%d\t%f\n" % (index1, index2, diff, weight))

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
        composition = FrozenGlycanComposition.parse(composition)
        node = CompositionGraphNode(composition, index, score)
        self.network.nodes.append(node)
        self.network.node_map[str(node.composition)] = node

    def handle_edge_line(self, line):
        index1, index2, diff, weight = line.replace("\n", "").split("\t")
        index1, index2, diff, weight = int(index1), int(index2), int(diff), float(weight)
        node1 = self.network.nodes[index1]
        node2 = self.network.nodes[index2]
        edge = CompositionGraphEdge(node1, node2, diff, weight)
        self.network.edges.append(edge)

    def handle_graph_file(self):
        state = "START"
        for line in self.file_obj:
            line = line[:-1]
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
