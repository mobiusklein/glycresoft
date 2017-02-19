from collections import defaultdict, deque
from operator import attrgetter
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


class EdgePath(object):
    def __init__(self, path=tuple()):
        self.path = tuple(path)
        self.members = frozenset(path)

    def __contains__(self, i):
        return i in self.members

    def add(self, i):
        extended = self.path + (i,)
        return EdgePath(extended)

    def __or__(self, other):
        return self.add(other)

    def __eq__(self, other):
        return self.members == other.members

    def __repr__(self):
        return "EdgePath%r" % (self.path,)

    def __iter__(self):
        return iter(self.path)


class DijkstraPathFinder(object):
    def __init__(self, graph, start, end, limit=float('inf')):
        self.graph = graph
        self.start = start
        self.end = end
        self.distance = defaultdict(lambda: float('inf'))
        self.distance[start._str] = 0
        self.unvisited_finite_distance = dict()
        self.limit = limit

    def find_smallest_unvisited(self, unvisited):
        smallest_distance = float('inf')
        smallest_node = None

        if not self.unvisited_finite_distance:
            iterable = [(k, self.distance[k]) for k in unvisited]
        else:
            iterable = self.unvisited_finite_distance.items()

        for node, distance in iterable:
            if distance <= smallest_distance:
                smallest_distance = distance
                smallest_node = node
        return self.graph[smallest_node]

    def find_path(self):
        unvisited = set([node._str for node in self.graph])

        visit_queue = deque([self.start])
        while self.end._str in unvisited:
            try:
                current_node = visit_queue.popleft()
            except IndexError:
                current_node = self.find_smallest_unvisited(unvisited)
            try:
                unvisited.remove(current_node._str)
            except KeyError:
                continue
            try:
                self.unvisited_finite_distance.pop(current_node._str)
            except KeyError:
                pass
            edges = current_node.edges
            for edge in edges:
                terminal = edge._traverse(current_node)
                if terminal._str not in unvisited:
                    continue
                path_length = self.distance[current_node._str] + edge.order
                terminal_distance = self.distance[terminal._str]
                if terminal_distance > path_length:
                    self.distance[terminal._str] = path_length
                    if terminal._str in unvisited and terminal_distance < self.limit:
                        self.unvisited_finite_distance[terminal._str] = path_length
                if terminal_distance < self.limit:
                    visit_queue.append(terminal)

    def search(self):
        self.find_path()
        return self.distance[self.end._str]


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
        self._composition = defaultdict(int, self.composition)
        self.index = index
        self.edges = EdgeSet()
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

    def edge_to(self, node):
        return self.edges.edge_to(self, node)

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


class EdgeSet(object):
    def __init__(self, store=None):
        if store is None:
            store = dict()
        self.store = store

    def edge_to(self, node1, node2):
        if node2.index < node1.index:
            node1, node2 = node2, node1
        return self.store[node1, node2]

    def add(self, edge):
        self.store[edge.node1, edge.node2] = edge

    def add_if_shorter(self, edge):
        try:
            prev = self.store[edge.node1, edge.node2]
            if prev.order < edge.order:
                return False
            else:
                self.store[edge.node1, edge.node2] = edge
                return True
        except KeyError:
            self.store[edge.node1, edge.node2] = edge
            return True

    def remove(self, edge):
        self.store.pop((edge.node1, edge.node2))

    def __iter__(self):
        return iter(self.store.values())

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return str(set(self.store.values()))

    def __eq__(self, other):
        return self.store == other.store


class CompositionGraphEdge(object):
    __slots__ = ["node1", "node2", "order", "weight", "_hash", "_str"]

    def __init__(self, node1, node2, order, weight=1.0):
        self.node1 = node1
        self.node2 = node2
        self.order = order if order > 0 else 1
        self.weight = weight
        self._hash = hash((node1, node2, order))
        self._str = "(%s)" % ', '.join(map(str, (node1, node2, order)))

        node1.edges.add_if_shorter(self)
        node2.edges.add_if_shorter(self)

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

    def __init__(self, compositions, distance_fn=n_glycan_distance):
        self.nodes = []
        self.node_map = {}
        self.distance_fn = distance_fn
        self.create_nodes(compositions)
        self.edges = EdgeSet()

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
        if distance_fn is None:
            distance_fn = self.distance_fn
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

    def remove_node(self, node, bridge=True, limit=5):
        subtracted_edges = list(node.edges)
        for edge in subtracted_edges:
            self.remove_edge(edge)
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
                    try:
                        if node1.edge_to(node2):
                            continue
                    except KeyError:
                        pass
                    path_finder = DijkstraPathFinder(self, node1, node2, min(old_distance + 1, limit))
                    shortest_path = path_finder.search()

                    # The distance function may not be transitive, in which case, an "edge through"
                    # solution may not make sense.
                    # shortest_path = self.distance_fn(node1._composition, node2._composition)[0]
                    path_length = min(old_distance, shortest_path)
                    if path_length < limit:
                        if node1.index > node2.index:
                            node1, node2 = node2, node1
                        new_edge = CompositionGraphEdge(node1, node2, path_length, 1.)
                        self.edges.add_if_shorter(new_edge)
        return subtracted_edges

    def remove_edge(self, edge):
        edge.remove()
        self.edges.remove(edge)

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
            except Exception as e:
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
        graph = CompositionGraph([], self.distance_fn)
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
