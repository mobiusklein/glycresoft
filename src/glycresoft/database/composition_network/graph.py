import numbers as abc_numbers
from io import StringIO
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from glypy.composition import composition_transform
from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, GlycanComposition


from glycopeptidepy import HashableGlycanComposition
from glycopeptidepy.structure.glycan import GlycanCompositionProxy

from glycresoft import symbolic_expression

from .space import (n_glycan_distance, composition_distance, CompositionSpace)
from .neighborhood import NeighborhoodCollection
from .rule import CompositionRuleClassifier


_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")


class CompositionNormalizer(object):
    cache: Dict[HashableGlycanComposition, HashableGlycanComposition]

    def __init__(self, cache=None):
        if cache is None:
            cache = dict()
        self.cache = cache

    def _normalize_key(self, key: Union[str, GlycanCompositionProxy, GlycanComposition]) -> HashableGlycanComposition:
        if isinstance(key, str):
            key = HashableGlycanComposition.parse(key)
        return HashableGlycanComposition(
            {self._normalize_monosaccharide(k): v for k, v in key.items()})

    def _normalize_monosaccharide(self, key):
        is_derivatized = composition_transform.has_derivatization(key)
        if is_derivatized:
            key = key.copy_underivatized()
        return key

    def _get_solution(self, key: Union[HashableGlycanComposition, str,
                                       GlycanComposition, GlycanCompositionProxy]) -> HashableGlycanComposition:
        if isinstance(key, (str, HashableGlycanComposition)):
            try:
                return self.cache[key]
            except KeyError:
                result = self._normalize_key(key)
                self.cache[key] = result
                return result
        else:
            return self._normalize_key(key)

    def normalize_composition(self, c):
        return self._get_solution(c)

    def __call__(self, c):
        return self.normalize_composition(c)

    def copy(self):
        return self.__class__(self.cache.copy())


normalize_composition = CompositionNormalizer()


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
                        self.unvisited_finite_distance[
                            terminal._str] = path_length
                if terminal_distance < self.limit:
                    visit_queue.append(terminal)

    def search(self):
        self.find_path()
        return self.distance[self.end._str]


try:
    _has_c = True
    from glycresoft._c.composition_network.graph import DijkstraPathFinder
except ImportError:
    _has_c = False

class CompositionGraphNode(object):
    composition: HashableGlycanComposition
    index: int
    _score: float
    marked: bool
    edges: 'EdgeSet'
    _str: str
    _hash: int
    internal_score: float

    __slots__ = (
        "composition", "index", "_score", "marked", "edges", "_str", "_hash", "internal_score")

    def __init__(self, composition, index, score=0., marked=False, **kwargs):
        self.composition = composition
        self.index = index
        self.edges = EdgeSet()
        self._str = str(self.composition)
        self._hash = hash(str(self._str))
        self._score = score
        self.internal_score = 0.0
        self.marked = marked

    @property
    def glycan_composition(self):
        return self.composition

    @property
    def order(self):
        return len(self.edges)

    @property
    def score(self):
        if self._score == 0:
            return self.internal_score
        else:
            return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def edge_to(self, node):
        return self.edges.edge_to(self, node)

    def neighbors(self):
        result = [edge._traverse(self) for edge in self.edges]
        return result

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
        return "CompositionGraphNode(%s, %d, %0.2f)" % (
            self._str, int(self.index) if self.index is not None else -1,
            self.score if self.score != 0 else self.internal_score)

    def copy(self):
        dup = CompositionGraphNode(self.composition, self.index, self.score)
        dup.internal_score = self.internal_score
        return dup

    def clone(self):
        return self.copy()


class EdgeSet(object):
    store: Dict[Tuple[CompositionGraphNode, CompositionGraphNode], 'CompositionGraphEdge']

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

    node1: CompositionGraphNode
    node2: CompositionGraphNode
    order: int
    weight: float
    _hash: int
    _str: str

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


class CompositionGraphBase(object):
    def copy(self):
        graph = CompositionGraph([], self.distance_fn)
        graph._composition_normalizer = self._composition_normalizer.copy()
        for node in self.nodes:
            graph.add_node(node.clone())
        for edge in self.edges:
            n1 = graph.nodes[edge.node1.index]
            n2 = graph.nodes[edge.node2.index]
            e = edge.copy_for(n1, n2)
            graph.edges.add(e)
        graph.neighborhoods.update(self.neighborhoods.copy())
        return graph


try:
    _has_c = True
    from glycresoft._c.composition_network.graph import (
        CompositionGraphEdge, CompositionGraphNode, EdgeSet, CompositionGraphBase)
except ImportError:
    _has_c = False


class CompositionGraph(CompositionGraphBase):
    neighborhoods: NeighborhoodCollection
    nodes: List[CompositionGraphNode]
    node_map: Dict[HashableGlycanComposition, CompositionGraphNode]
    _composition_normalizer: CompositionNormalizer
    distance_fn: Callable[[HashableGlycanComposition, HashableGlycanComposition], Tuple[float, float]]
    edges: EdgeSet
    cache_state: Dict

    def __init__(self, compositions, distance_fn=n_glycan_distance, neighborhoods=None, cache_state: Optional[Dict]=None):
        if neighborhoods is None:
            neighborhoods = []
        if cache_state is None:
            cache_state = {}
        self.nodes = []
        self.node_map = {}
        self._composition_normalizer = CompositionNormalizer()
        self.distance_fn = distance_fn
        self.create_nodes(compositions)
        self.edges = EdgeSet()
        self.neighborhoods = NeighborhoodCollection(neighborhoods)
        self.cache_state = cache_state

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
        compositions = map(self.normalize_composition, compositions)

        for c in sorted(set(compositions), key=lambda x: (x.mass(), len(x))):
            n = CompositionGraphNode(c, i)
            self.add_node(n)
            i += 1

    def add_node(self, node: CompositionGraphNode, reindex=False):
        """Given a CompositionGraphNode, add it to the graph.

        If `reindex` is `True` then a full re-indexing of the graph will
        take place after the insertion is made. Otherwise it is assumed that
        `node` is being added in index-specified order.

        Parameters
        ----------
        node : CompositionGraphNode
            The node to be added
        """
        key = str(self.normalize_composition(node.composition))
        if key in self.node_map:
            raise ValueError("Redundant composition!")
        self.nodes.append(node)
        self.node_map[key] = node

        if reindex:
            self._reindex()
        return self

    def create_edges(self, distance=1, distance_fn=composition_distance):
        """Traverse composition-space to find nodes similar to each other
        and construct CompositionGraphEdges between them to represent that
        similarity.

        Parameters
        ----------
        distance : int, optional
            The maximum dissimilarity between two nodes permitted
            to allow the construction of an edge between them
        distance_fn : callable, optional
            A function to use to compute the bounded distance between two nodes
            that are within `distance` of eachother in raw composition-space
        """
        if distance_fn is None:
            distance_fn = self.distance_fn
        space = CompositionSpace([node.composition for node in self])
        for node in self:
            if node is None:
                continue
            for related in space.find_related(node.composition, distance):
                related_node = self.node_map[str(related)]
                # Ensures no duplicates assuming symmetric search window
                if node.index < related_node.index:
                    diff, weight = distance_fn(node.composition, related)
                    e = CompositionGraphEdge(node, related_node, diff, weight)
                    self.edges.add(e)
        return self

    def add_edge(self, node1: CompositionGraphNode, node2: CompositionGraphNode):
        if node1.index > node2.index:
            node1, node2 = node2, node1
        diff, weight = self.distance_fn(node1.composition, node2.composition)
        e = CompositionGraphEdge(node1, node2, diff, weight)
        self.edges.add(e)

    def remove_node(self, node: CompositionGraphNode, bridge: bool=True, limit: int=5, ignore_marked: bool=True):
        """Removes the Glycan Composition given by `node` from the graph
        and all edges connecting to it. This will reindex the graph.

        If two Glycan Compositions `x` and `y` are connected *through* `node`,
        `bridge` is true,  and the shortest path connecting `x` and
        `y` is longer than the sum of the path from `x` to `node` and from `node`
        to `y`, create a new edge connecting `x` and `y`, if the new edge is shorter
        than `limit`.

        Parameters
        ----------
        node : CompositionGraphNode-like
            The node to be removed
        bridge : bool, optional
            Whether or not to create new edges
            bridging neighbors
        limit : int, optional
            The maximum path length under which
            to bridge neighbors

        Returns
        -------
        list
            The list of edges removed
        """
        node = self[node.glycan_composition]
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
                    if ignore_marked and (node1.marked or node2.marked):
                        continue
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
                    path_finder = DijkstraPathFinder(
                        self, node1, node2, min(old_distance + 1, limit))
                    shortest_path = path_finder.search()

                    # The distance function may not be transitive, in which case, an "edge through"
                    # solution may not make sense.
                    # shortest_path = self.distance_fn(node1._composition, node2._composition)[0]
                    path_length = min(old_distance, shortest_path)
                    if path_length < limit:
                        if node1.index > node2.index:
                            node1, node2 = node2, node1
                        new_edge = CompositionGraphEdge(
                            node1, node2, path_length, 1.)
                        self.edges.add_if_shorter(new_edge)
        return subtracted_edges

    def remove_edge(self, edge: CompositionGraphEdge):
        edge.remove()
        self.edges.remove(edge)

    def _reindex(self):
        i = 0
        for node in self:
            node.index = i
            i += 1

    def normalize_composition(self, composition):
        return self._composition_normalizer(composition)

    def __getitem__(self, key):
        if isinstance(key, (str, GlycanComposition, GlycanCompositionProxy)):
            key = self.normalize_composition(key)
            return self.node_map[key]
        # use the ABC Integral to catch all numerical types that could be used as an
        # index including builtin int and long, as well as all the NumPy flavors of
        # integers
        elif isinstance(key, (abc_numbers.Integral, slice)):
            return self.nodes[key]
        else:
            try:
                result = []
                iter(key)
                try:
                    if isinstance(key[0], (bool, np.bool_)):
                        for b, k in zip(key, self):
                            if b:
                                result.append(k)
                    else:
                        for k in key:
                            result.append(self[k])
                except TypeError:
                    # Handle unindexable iteratables
                    for k in key:
                        result.append(self[k])
                return result
            except Exception as e:
                if len(key) == 0:
                    return []
                raise IndexError(
                    "An error occurred (%r) during indexing with %r" % (e, key))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __reduce__(self):
        return self.__class__, ([], self.distance_fn), self.__getstate__()

    def __getstate__(self):
        string_buffer = StringIO()
        dump(self, string_buffer)
        return string_buffer, self.distance_fn

    def __setstate__(self, state):
        string_buffer, distance_fn = state
        string_buffer.seek(0)
        net, neighborhoods = load(string_buffer)
        self.nodes = net.nodes
        self.neighborhoods = NeighborhoodCollection(neighborhoods)
        self.node_map = net.node_map
        self.edges = net.edges
        self.distance_fn = distance_fn
        self._composition_normalizer = CompositionNormalizer()

    def __repr__(self):
        return "{self.__class__.__name__}({node_count} nodes, {edge_count} edges)".format(
            self=self, node_count=len(self), edge_count=len(self.edges))

    def clone(self):
        return self.copy()

    def __eq__(self, other):
        try:
            return self.nodes == other.nodes and self.edges == other.edges
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)

    def assign(self, observed, inplace=False):
        if not inplace:
            network = self.clone()
        else:
            network = self
        solution_map = {}
        for case in observed:
            if case.glycan_composition is None:
                continue
            s = solution_map.get(case.glycan_composition)
            if s is None or s.score < case.score:
                solution_map[case.glycan_composition] = case

        for node in network.nodes:
            node.internal_score = 0

        for composition, solution in solution_map.items():
            try:
                node = network[composition]
                node.internal_score = solution.internal_score
            except KeyError:
                # Not all exact compositions have nodes
                continue
        return network

    def unassign(self, inplace=False):
        if not inplace:
            network = self.clone()
        else:
            network = self
        for node in network:
            node.score = node.internal_score = 0
        return network

    def augment_with_decoys(self, pseudodistance=2):
        compositions = []
        if pseudodistance == 0:
            raise ValueError("Cannot use decoys with pseudodistance of size 0")
        for node in self:
            t = node.glycan_composition.copy()
            d = node.glycan_composition.copy()
            d['#decoy#'] = pseudodistance
            compositions.append(t)
            compositions.append(d)
        return self.__class__(compositions, self.distance_fn, self.neighborhoods)

    def merge(self, other):
        edges = other.edges
        for node in other:
            self.add_node(node)
        self._reindex()
        for edge in edges:
            self.add_edge(self[edge.node1], self[edge.node2])
        return self


try:
    _has_c = True
    from glycresoft._c.composition_network.graph import reindex_graph as _reindex_graph
    CompositionGraph._reindex = _reindex_graph
except ImportError:
    _has_c = False


class GraphWriter(object):

    def __init__(self, network, file_obj):
        self.network = network
        self.file_obj = file_obj
        self.handle_graph(self.network)

    def write(self, text):
        self.file_obj.write(text)

    def handle_node(self, node):
        composition = node.composition.serialize()
        index = (node.index)
        score = node.score
        self.write("%d\t%s\t%f\n" % (index, composition, score))

    def handle_edge(self, edge):
        index1 = edge.node1.index
        index2 = edge.node2.index
        diff = edge.order
        weight = edge.weight
        line = ("%d\t%d\t%d\t%f" % (index1, index2, diff, weight))
        trimmed = line.rstrip("0")
        if line != trimmed:
            trimmed += '0'
        self.write(trimmed + '\n')

    def handle_graph(self, graph):
        self.write("#COMPOSITIONGRAPH 1.1\n")
        self.write("#NODE\n")
        for node in self.network.nodes:
            self.handle_node(node)
        self.write("#EDGE\n")
        for edge in self.network.edges:
            self.handle_edge(edge)
        try:
            if graph.neighborhoods:
                self.write("#NEIGHBORHOOD\n")
                for neighborhood in graph.neighborhoods:
                    self.handle_neighborhood(neighborhood)
        except AttributeError:
            pass

    def handle_neighborhood(self, neighborhood_rule):
        self.write("BEGIN NEIGHBORHOOD\n")
        self.write(neighborhood_rule.serialize())
        self.write("END NEIGHBORHOOD\n")


class GraphReader(object):

    @classmethod
    def read(cls, file_obj):
        inst = cls(file_obj)
        return inst.network, inst.neighborhoods

    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.network = CompositionGraph([])
        self.neighborhoods = self.network.neighborhoods
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
        node.internal_score = score
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
        buffering = []
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
                if line == "#NEIGHBORHOOD":
                    state = "NEIGHBORHOOD"
                else:
                    self.handle_edge_line(line)
            elif state == "NEIGHBORHOOD":
                if line.startswith("BEGIN NEIGHBORHOOD"):
                    if buffering:
                        self.handle_neighborhood(buffering)
                    buffering = []
                elif line.startswith("END NEIGHBORHOOD"):
                    if buffering:
                        self.handle_neighborhood(buffering)
                        buffering = []
                else:
                    buffering.append(line)

    def handle_neighborhood(self, lines):
        rule = CompositionRuleClassifier.parse(lines)
        self.neighborhoods.add(rule)


dump = GraphWriter
load = GraphReader.read
