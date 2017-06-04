from collections import defaultdict, deque, OrderedDict
import numbers as abc_numbers
import numpy as np

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    basestring
except NameError:
    basestring = (str, bytes)

from glypy.composition.glycan_composition import FrozenMonosaccharideResidue, GlycanComposition


from glycopeptidepy import HashableGlycanComposition
from glycopeptidepy.structure.glycan import GlycanCompositionProxy
from glycopeptidepy.utils import simple_repr


from .glycan_composition_filter import GlycanCompositionFilter
from .. import symbolic_expression


_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")


def composition_distance(c1, c2):
    '''N-Dimensional Manhattan Distance or L1 Norm
    '''
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


class CompositionNormalizer(object):
    def __init__(self, cache=None):
        if cache is None:
            cache = dict()
        self.cache = cache

    def _normalize_key(self, key):
        if isinstance(key, basestring):
            key = HashableGlycanComposition.parse(key)
        key = symbolic_expression.GlycanSymbolContext(key)
        return HashableGlycanComposition.parse(key.serialize())

    def _get_solution(self, key):
        if isinstance(key, (basestring, HashableGlycanComposition)):
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
        self.edges = EdgeSet()
        self._str = str(self.composition)
        self._hash = hash(str(self._str))
        self._score = score
        self.internal_score = 0.0

    @property
    def glycan_composition(self):
        return self.composition

    @property
    def order(self):
        return len(self.edges)

    @property
    def score(self):
        if self._score == 0:
            return self._temp_score
        else:
            return self._score

    @score.setter
    def score(self, value):
        self._score = value

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
        return "CompositionGraphNode(%s, %d, %0.2f)" % (
            self._str, len(self.edges),
            self.score if self.score != 0 else self._temp_score)

    def clone(self):
        dup = CompositionGraphNode(self.composition, self.index, self.score)
        dup.internal_score = self.internal_score
        return dup


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
        self._composition_normalizer = CompositionNormalizer()
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
        compositions = map(self._composition_normalizer, compositions)

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
            key = self._composition_normalizer(key)
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

    def handle_graph(self, graph, neighborhoods=None):
        self.file_obj.write("#COMPOSITIONGRAPH 1.0\n")
        self.file_obj.write("#NODE\n")
        for node in self.network.nodes:
            self.handle_node(node)
        self.file_obj.write("#EDGE\n")
        for edge in self.network.edges:
            self.handle_edge(edge)

        if neighborhoods is not None:
            for neighborhood in neighborhoods:
                self.handle_neighborhood(neighborhoods)

    def handle_neighborhood(self, neighborhood_rule):
        self.file_obj.write("BEGIN NEIGHBORHOOD\n")
        self.file_obj.write(neighborhood_rule.serialize())
        self.file_obj.write("END NEIGHBORHOOD\n")


class GraphReader(object):

    @classmethod
    def read(cls, file_obj):
        return cls(file_obj).network

    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.network = CompositionGraph([])
        self.neighborhoods = NeighborhoodCollection()
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


class CompositionRuleBase(object):

    __repr__ = simple_repr

    def get_composition(self, obj):
        try:
            composition = obj.glycan_composition
        except AttributeError:
            composition = HashableGlycanComposition.parse(obj)
        composition = symbolic_expression.GlycanSymbolContext(composition)
        return composition

    def __and__(self, other):
        if isinstance(other, CompositionRuleClassifier):
            other = other.copy()
            other.rules.append(self)
            return other
        else:
            return CompositionRuleClassifier(None, [self, other])

    def get_symbols(self):
        raise NotImplementedError()

    @property
    def symbols(self):
        return self.get_symbols()

    def is_univariate(self):
        return len(self.get_symbols()) == 1

    def serialize(self):
        raise NotImplementedError()

    @classmethod
    def parse(cls, line, handle=None):
        raise NotImplementedError()


def int_or_none(x):
    try:
        return int(x)
    except ValueError:
        return None


class CompositionExpressionRule(CompositionRuleBase):
    def __init__(self, expression, required=True):
        self.expression = symbolic_expression.ExpressionNode.parse(str(expression))
        self.required = required

    def get_symbols(self):
        return self.expression.get_symbols()

    def __call__(self, obj):
        composition = self.get_composition(obj)
        if composition.partially_defined(self.expression):
            return composition[self.expression]
        else:
            if self.required:
                return False
            else:
                return True

    def serialize(self):
        tokens = ["CompositionExpressionRule", str(self.expression),
                  str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionExpressionRule" and i < n:
            i += 1
        i += 1
        if i >= n:
            raise ValueError("Coult not parse %r with %s" % (line, cls))
        expr = symbolic_expression.parse_expression(tokens[i])
        required = tokens[i + 1].lower() == 'true'
        return cls(expr, required)


class CompositionRangeRule(CompositionRuleBase):

    def __init__(self, expression, low=None, high=None, required=True):
        self.expression = symbolic_expression.ExpressionNode.parse(str(expression))
        self.low = low
        self.high = high
        self.required = required

    def get_symbols(self):
        return self.expression.get_symbols()

    def __call__(self, obj):
        composition = self.get_composition(obj)
        if composition.partially_defined(self.expression):
            if self.low is None:
                return composition[self.expression] <= self.high
            elif self.high is None:
                return self.low <= composition[self.expression]
            return self.low <= composition[self.expression] <= self.high
        else:
            if self.required and self.low > 0:
                return False
            else:
                return True

    def serialize(self):
        tokens = ["CompositionRangeRule", str(self.expression), str(self.low),
                  str(self.high), str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionRangeRule" and i < n:
            i += 1
        i += 1
        if i >= n:
            raise ValueError("Coult not parse %r with %s" % (line, cls))
        expr = symbolic_expression.parse_expression(tokens[i])
        low = int_or_none(tokens[i + 1])
        high = int_or_none(tokens[i + 2])
        required = tokens[i + 3].lower() == 'true'
        return cls(expr, low, high, required)


class CompositionRatioRule(CompositionRuleBase):
    def __init__(self, numerator, denominator, ratio_threshold, required=True):
        self.numerator = numerator
        self.denominator = denominator
        self.ratio_threshold = ratio_threshold
        self.required = required

    def _test(self, x):
        if isinstance(self.ratio_threshold, (tuple, list)):
            return self.ratio_threshold[0] <= x < self.ratio_threshold[1]
        else:
            return x >= self.ratio_threshold

    def get_symbols(self):
        return (self.numerator, self.denominator)

    def __call__(self, obj):
        composition = self.get_composition(obj)
        val = composition[self.numerator]
        ref = composition[self.denominator]

        if ref == 0 and self.required:
            return False
        else:
            ratio = val / float(ref)
            return self._test(ratio)

    def serialize(self):
        tokens = ["CompositionRatioRule", str(self.numerator), str(self.denominator),
                  str(self.ratio_threshold), str(self.required)]
        return '\t'.join(tokens)

    @classmethod
    def parse(cls, line, handle=None):
        tokens = line.strip().split("\t")
        n = len(tokens)
        i = 0
        while tokens[i] != "CompositionRatioRule" and i < n:
            i += 1
        i += 1
        numerator = symbolic_expression.parse_expression(tokens[i])
        denominator = symbolic_expression.parse_expression(tokens[i + 1])
        ratio_threshold = float(tokens[i + 2])
        required = tokens[i + 3].lower() == 'true'
        return cls(numerator, denominator, ratio_threshold, required)


class CompositionRuleClassifier(object):

    def __init__(self, name, rules):
        self.name = name
        self.rules = rules

    def __iter__(self):
        return iter(self.rules)

    def __call__(self, obj):
        for rule in self:
            if not rule(obj):
                return False
        return True

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            return self.name == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    __repr__ = simple_repr

    def copy(self):
        return CompositionRuleClassifier(self.name, list(self.rules))

    def __and__(self, other):
        if isinstance(other, CompositionRuleClassifier):
            other = other.copy()
            other.rules.extend(self.rules)
            return other
        else:
            self = self.copy()
            self.rules.append(other)
            return self

    def get_symbols(self):
        symbols = set()
        for rule in self:
            symbols.update(rule.symbols)
        return symbols

    @property
    def symbols(self):
        return self.get_symbols()

    def serialize(self):
        text_buffer = StringIO()
        text_buffer.write("CompositionRuleClassifier\t%s\n" % (self.name,))
        for rule in self:
            text = rule.serialize()
            text_buffer.write("\t%s\n" % (text,))
        text_buffer.seek(0)
        return text_buffer.read()

    @classmethod
    def parse(cls, lines):
        line = lines[0]
        name = line.strip().split("\t")[1]
        rules = []
        for line in lines[1:]:
            if line == "":
                continue
            rule_type = line.split("\t")[1]
            if rule_type == "CompositionRangeRule":
                rule = CompositionRangeRule.parse(line)
            elif rule_type == "CompositionRatioRule":
                rule = CompositionRatioRule.parse(line)
            elif rule_type == "CompositionExpressionRule":
                rule = CompositionExpressionRule.parse(line)
            else:
                raise ValueError("Unrecognized Rule Type: %r" % (line,))
            rules.append(rule)
        return cls(name, rules)


class NeighborhoodCollection(object):
    def __init__(self, neighborhoods=None):
        if neighborhoods is None:
            neighborhoods = OrderedDict()
        self.neighborhoods = OrderedDict()
        if isinstance(neighborhoods, (dict)):
            self.neighborhoods = OrderedDict(neighborhoods)
        else:
            for item in neighborhoods:
                self.add(item)

    def add(self, classifier):
        self.neighborhoods[classifier.name] = classifier

    def __iter__(self):
        return iter(self.neighborhoods.values())

    def __repr__(self):
        return "NeighborhoodCollection(%s)" % (', '.join(self.neighborhoods.keys()))

    def get_neighborhood(self, key):
        return self.neighborhoods[key]

    def __getitem__(self, key):
        try:
            return self.get_neighborhood(key)
        except KeyError:
            if isinstance(key, abc_numbers.Number):
                return self.neighborhoods.values()[key]
            else:
                raise

    def __len__(self):
        return len(self.neighborhoods)


def make_n_glycan_neighborhoods():
    neighborhoods = NeighborhoodCollection()

    _neuraminic = "(%s)" % ' + '.join(map(str, (
        FrozenMonosaccharideResidue.from_iupac_lite("NeuAc"),
        FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")
    )))
    _hexose = "(%s)" % ' + '.join(
        map(str, map(FrozenMonosaccharideResidue.from_iupac_lite, ['Hex', ])))
    _hexnac = "(%s)" % ' + '.join(
        map(str, map(FrozenMonosaccharideResidue.from_iupac_lite, ['HexNAc', ])))

    high_mannose = CompositionRangeRule(
        _hexose, 3, 12) & CompositionRangeRule(
        _hexnac, 2, 2) & CompositionRangeRule(
        _neuraminic, 0, 0)
    high_mannose.name = "high-mannose"
    neighborhoods.add(high_mannose)

    # over_extended = CompositionRangeRule("%s - %s" % (
    #     _hexose,
    #     _hexnac), 3) & CompositionRangeRule(
    #     _neuraminic, 1, None)
    # over_extended.name = 'over-extended'
    # neighborhoods.add(over_extended)

    base_hexnac = 3
    base_neuac = 2
    for i, spec in enumerate(['hybrid', 'bi', 'tri', 'tetra', 'penta']):
        if i == 0:
            rule = CompositionRangeRule(
                _hexnac, base_hexnac - 1, base_hexnac + 1
            ) & CompositionRangeRule(
                _neuraminic, 0, base_neuac) & CompositionRangeRule(
                _hexose, base_hexnac + i - 1,
                base_hexnac + i + 3)
            rule.name = spec
            neighborhoods.add(rule)
        else:
            sialo = CompositionRangeRule(
                _hexnac, base_hexnac + i - 1, base_hexnac + i + 1
            ) & CompositionRangeRule(
                _neuraminic, 1, base_neuac + i
            ) & CompositionRangeRule(
                _hexose, base_hexnac + i - 1,
                base_hexnac + i + 2)

            sialo.name = "%s-antennary" % spec
            asialo = CompositionRangeRule(
                _hexnac, base_hexnac + i - 1, base_hexnac + i + 1
            ) & CompositionRangeRule(
                _neuraminic, 0, 1
            ) & CompositionRangeRule(
                _hexose, base_hexnac + i - 1,
                base_hexnac + i + 2)

            asialo.name = "asialo-%s-antennary" % spec
            neighborhoods.add(sialo)
            neighborhoods.add(asialo)
    return neighborhoods


_n_glycan_neighborhoods = make_n_glycan_neighborhoods()


class NeighborhoodWalker(object):

    def __init__(self, network, neighborhoods=None):
        if neighborhoods is None:
            neighborhoods = NeighborhoodCollection(_n_glycan_neighborhoods)
        self.network = network
        self.neighborhood_assignments = defaultdict(set)
        self.neighborhoods = neighborhoods
        self.filter_space = GlycanCompositionFilter(
            [node.composition for node in self.network])

        self.symbols = symbolic_expression.SymbolSpace(self.filter_space.monosaccharides)

        self.neighborhood_maps = defaultdict(list)
        self.assign()

    def neighborhood_names(self):
        return [n.name for n in self.neighborhoods]

    def __getitem__(self, key):
        return self.neighborhood_assignments[key]

    def query_neighborhood(self, neighborhood):
        query = None
        filters = []
        for rule in neighborhood.rules:
            if not self.symbols.partially_defined(rule.symbols):
                continue

            filters.append(rule)
            try:
                low = rule.low
                high = rule.high
            except AttributeError:
                continue
            if low is None:
                low = 0
            if high is None:
                # No glycan will have more than 100 of a single residue
                # in practice.
                high = 100
            name = rule.symbols[0]
            if query is None:
                query = self.filter_space.query(name, low, high)
            else:
                query.add(name, low, high)
        if filters:
            query = filter(lambda x: all([f(x) for f in filters]), query)
        else:
            query = query.all()
        return query

    def assign(self):
        for neighborhood in self.neighborhoods:
            query = self.query_neighborhood(neighborhood)
            if query is None:
                print(neighborhood, self.filter_space)
                raise ValueError()
            for composition in query:
                if neighborhood(composition):
                    self.neighborhood_assignments[
                        self.network[composition]].add(neighborhood.name)
        for node in self.network:
            for neighborhood in self[node]:
                self.neighborhood_maps[neighborhood].append(node)

    def compute_belongingness(self, node, neighborhood, distance_fn=n_glycan_distance):
        count = 0
        total_weight = 0
        for member in self.neighborhood_maps[neighborhood]:
            distance, weight = distance_fn(node.glycan_composition, member.glycan_composition)
            if distance > 0:
                weight *= (1. / distance)
            else:
                weight = 1.0
            total_weight += weight
            count += 1
        if count == 0:
            return 0
        return total_weight / count
