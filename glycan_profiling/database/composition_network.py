from collections import defaultdict
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


class ShortestPathFinder(object):
    """A recursive algorithm for local graph exploration that finds
    the shortest path between two nodes in a graph.

    Analogous to a bounded version of Dijkstra's algorithm.

    Attributes
    ----------
    graph : CompositionGraph
        The graph to traverse
    solution_table : defaultdict(dict)
        A lookup table to memoize queries in
    """

    def __init__(self, graph):
        self.graph = graph
        self.solution_table = defaultdict(lambda: defaultdict(lambda: -1))

    def solve_path(self, start, end, limit=100, total=0, visited=None):
        if visited is None:
            visited = frozenset()
        dist = self.get_path_length(start, end)
        if dist > -1:
            return dist
        else:
            path_length = self._solve_path(start, end, limit, total, visited)
            self.set_path_length(start, end, path_length)
            return path_length

    def _path_length(self, accumulator, edge):
        return accumulator + edge.order

    def set_path_length(self, start, end, path_length):
        self.solution_table[start._str][end._str] = path_length
        self.solution_table[end._str][start._str] = path_length

    def get_path_length(self, start, end):
        return self.solution_table[start._str][end._str]

    def _solve_path(self, start, end, limit=0, total=0, visited=None):
        if visited is None:
            visited = frozenset()
        dist = self.get_path_length(start, end)
        if dist > -1:
            return dist
        else:
            if start is end:
                return total
            else:
                shortest_path = float('inf')
                for edge in start.edges:
                    terminal = edge._traverse(start)
                    if terminal in visited:
                        continue
                    new_total = self._path_length(total, edge)
                    self.set_path_length(start, terminal, new_total)
                    if limit > 0 and new_total > limit:
                        continue
                    out = self.solve_path(
                        terminal, end, limit, new_total, visited | {start._str})
                    if out < shortest_path:
                        shortest_path = out
                self.set_path_length(start, end, shortest_path)
                return shortest_path


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
        try:
            return (self)._str == str(other)
        except AttributeError:
            return str(self) == str(other)

    def __str__(self):
        return self._str

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return "CompositionGraphNode(%s, %d, %0.2e)" % (
            self._str, len(self.edges),
            self.score if self.score != 0 else self._temp_score)

    def clone(self):
        return CompositionGraphNode(self.composition, self.index, self.score)


class CompositionGraphEdge(object):
    __slots__ = ["node1", "node2", "order", "weight", "_hash", "_str"]

    def __init__(self, node1, node2, order, weight=1.0):
        self.node1 = node1
        self.node2 = node2
        self.order = order if order > 0 else 1
        self.weight = weight
        self._hash = hash((node1, node2, order))
        self._str = "(%s)" % ', '.join(map(str, (node1, node2, order)))

        node1.edges.add(self)
        node2.edges.add(self)

    def __getitem__(self, key):
        str_key = str(key)
        if str_key == self.node1._str:
            return self.node2
        elif str_key == self.node2._str:
            return self.node1
        else:
            raise KeyError(key)

    def _traverse(self, node):
        return self.node1 if node is self.node2 else self.node2

    def __eq__(self, other):
        return self._str == other._str

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

    def remove(self):
        try:
            self.node1.edges.remove(self)
        except KeyError:
            pass
        try:
            self.node2.edges.remove(self)
        except KeyError:
            pass


class CompositionGraph(object):

    def __init__(self, compositions):
        self.nodes = []
        self.node_map = {}
        self.create_nodes(compositions)
        self.edges = set()
        self.path_finder = ShortestPathFinder(self)

    def create_nodes(self, compositions):
        """Given an iterable of GlycanComposition-like or strings encoding GlycanComposition-like
        objects construct nodes representing these compositions and add them to the graph.

        The order items appear in the list will affect their graph node's :attr:`index` attribute
        and in turn affect their order in edges. For consistent behavior, order nodes by mass.

        Parameters
        ----------
        compositions : Iterable
            An iterable source of GlycanComposition-like objects
        """
        i = 0
        compositions = map(HashableGlycanComposition.parse, compositions)

        for c in compositions:
            n = CompositionGraphNode(c, i)
            self.add_node(n)
            i += 1

    def add_node(self, node, reindex=False):
        """Given a CompositionGraphNode, add it to the graph.

        If `reindex` is `True` then a full re-indexing of the graph will
        take place after the insertion is made. Otherwise it is assumed that
        `node` is being added in index-specified order.

        Parameters
        ----------
        node : CompositionGraphNode
            The node to be added
        """
        self.nodes.append(node)
        self.node_map[str(node.composition)] = node

        if reindex:
            self._reindex()

    def create_edges(self, degree=2, distance_fn=composition_distance):
        """Traverse composition-space to find nodes similar to each other
        and construct CompositionGraphEdges between them to represent that
        similarity.

        Parameters
        ----------
        degree : int, optional
            The maximum dissimilarity between two nodes permitted
            to allow the construction of an edge between them
        distance_fn : callable, optional
            A function to use to compute the bounded distance between two nodes
            that are within `degree` of eachother in raw composition-space
        """
        space = CompositionSpace([node.composition for node in self])
        for node in self:
            if node is None:
                continue
            for related in space.find_narrowly_related(node.composition, degree):
                related_node = self.node_map[str(related)]
                # Ensures no duplicates assuming symmetric search window
                if node.index < related_node.index:
                    diff, weight = distance_fn(node.composition, related)
                    e = CompositionGraphEdge(node, related_node, diff, weight)
                    self.edges.add(e)

    def remove_node(self, node, bridge=True, limit=20):
        subtracted_edges = list(node.edges)
        for edge in subtracted_edges:
            edge.remove()
            self.edges.remove(edge)
        self.nodes.pop(node.index)
        self.node_map.pop(str(node.composition))
        self._reindex()

        if bridge:
            seen = set()
            for edge in subtracted_edges:
                for other_edge in subtracted_edges:
                    if edge == other_edge:
                        continue
                    node1 = edge._traverse(node)
                    node2 = other_edge._traverse(node)
                    key = frozenset((node1.index, node2.index))
                    if key in seen:
                        continue
                    seen.add(key)
                    old_distance = edge.order + other_edge.order
                    if old_distance >= limit:
                        continue
                    shortest_path = self.path_finder.solve_path(node1, node2, min(old_distance + 1, limit))
                    if shortest_path >= old_distance:
                        if node1.index > node2.index:
                            node1, node2 = node2, node1
                        new_edge = CompositionGraphEdge(node1, node2, old_distance, 1.)
                        self.edges.add(new_edge)
        return subtracted_edges

    def _reindex(self):
        i = 0
        for node in self:
            node.index = i
            i += 1

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
            graph.edges.add(e)
        return graph

    def __eq__(self, other):
        try:
            return self.nodes == other.nodes and self.edges == other.edges
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)


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
        self.network.edges.add(edge)

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
