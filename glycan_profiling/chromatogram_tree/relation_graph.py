import numpy as np

from collections import deque, defaultdict
import itertools

from .grouping import ChromatogramRetentionTimeInterval, IntervalTreeNode
from ms_deisotope.peak_dependency_network.intervals import SpanningMixin
from .index import ChromatogramFilter

from glypy.composition.glycan_composition import FrozenMonosaccharideResidue

_standard_transitions = [
    FrozenMonosaccharideResidue.from_iupac_lite("HexNAc"),
    FrozenMonosaccharideResidue.from_iupac_lite("Hex"),
    FrozenMonosaccharideResidue.from_iupac_lite('NeuAc'),
    FrozenMonosaccharideResidue.from_iupac_lite("Fuc"),
    FrozenMonosaccharideResidue.from_iupac_lite("HexA"),
]


class UnknownTransition(object):
    def __init__(self, mass):
        self._mass = mass
        self._hash = hash(mass)

    def __eq__(self, other):
        return self.mass() == other.mass()

    def mass(self):
        return self._mass

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return "{self.__class__.__name__}({mass})".format(self=self, mass=self.mass())


class TimeQuery(SpanningMixin):
    def __init__(self, chromatogram, width=0):
        self.start = chromatogram.start_time - width
        self.end = chromatogram.end_time + width

    def __repr__(self):
        return "TimeQuery(%f, %f)" % (self.start, self.end)


class ChromatogramGraphNode(ChromatogramRetentionTimeInterval):
    def __init__(self, chromatogram, index, edges=None):
        super(ChromatogramGraphNode, self).__init__(chromatogram)
        if edges is None:
            edges = set()
        self.chromatogram = chromatogram
        self.index = index
        self.edges = edges

        total = 0
        abundance = 0
        for node in self.chromatogram.nodes:
            intensity = node.total_intensity()
            total += node.retention_time * intensity
            abundance += intensity
        self.center = total / abundance

    def get_chromatogram(self):
        return self.chromatogram

    def __repr__(self):
        return "ChromatogramGraphNode(%s)" % (self.chromatogram,)

    def __index__(self):
        return self.index

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.chromatogram == other.chromatogram


class ChromatogramGraphEdge(object):
    def __init__(self, node_a, node_b, transition, weight=1, mass_error=0, rt_error=0):
        self.node_a = node_a
        self.node_b = node_b
        self.transition = transition
        self.weight = weight
        self.mass_error = mass_error
        self.rt_error = rt_error
        self.node_a.edges.add(self)
        self.node_b.edges.add(self)

    def __repr__(self):
        return "ChromatogramGraphEdge(%s, %s, %s)" % (
            self.node_a.chromatogram, self.node_b.chromatogram, self.transition)

    def __hash__(self):
        return hash(frozenset((self.node_a.index, self.node_b.index)))

    def __eq__(self, other):
        return frozenset((self.node_a.index, self.node_b.index)) == frozenset((other.node_a.index, other.node_b.index))


def explode_node(node):
    items = deque()
    results = []

    for edge in node.edges:
        items.append(edge.node_a)
        items.append(edge.node_b)

    visited = set()
    while items:
        node = items.popleft()
        if node.index in visited:
            continue
        visited.add(node.index)
        results.append(node)

        for edge in node.edges:
            if edge.node_a.index not in visited:
                items.append(edge.node_a)
            if edge.node_b.index not in visited:
                items.append(edge.node_b)
    return results


def chromatograms_from_edge(edge):
    return edge.node_a.chromatogram, edge.node_b.chromatogram


class ChromatogramGraph(object):
    def __init__(self, chromatograms):
        self.chromatograms = chromatograms
        self.assigned_seed_queue = deque()
        self.assignment_map = {}
        self.nodes = self._construct_graph_nodes(self.chromatograms)
        self.rt_tree = IntervalTreeNode.build(self.nodes)
        self.edges = set()

    def __len__(self):
        return len(self.nodes)

    def _construct_graph_nodes(self, chromatograms):
        nodes = []
        for i, chroma in enumerate(chromatograms):
            node = (ChromatogramGraphNode(chroma, i))
            nodes.append(node)
            if node.chromatogram.composition:
                self.enqueue_seed(node)
                self.assignment_map[node.chromatogram.composition] = node
        return nodes

    def enqueue_seed(self, chromatogram_with_assignment):
        self.assigned_seed_queue.append(chromatogram_with_assignment)

    def pop_seed(self):
        return self.assigned_seed_queue.popleft()

    def iterseeds(self):
        while self.assigned_seed_queue:
            yield self.pop_seed()

    def find_edges(self, node, query_width=2., transitions=None, **kwargs):
        if transitions is None:
            transitions = _standard_transitions
        query = TimeQuery(node.chromatogram, query_width)
        nodes = ChromatogramFilter(self.rt_tree.overlaps(query.start, query.end))

        starting_mass = node.neutral_mass

        for transition in transitions:
            added = starting_mass + transition.mass()
            match = nodes.find_mass(added)
            if match:
                ppm_error = (added - match.neutral_mass) / match.neutral_mass
                rt_error = (node.center - match.center)
                self.edges.add(ChromatogramGraphEdge(node, match, transition, mass_error=ppm_error, rt_error=rt_error))

    def find_shared_peaks(self):
        peak_map = defaultdict(set)
        for chromatogram_node in self.nodes:
            for peak_bunch in chromatogram_node.chromatogram.peaks:
                for peak in peak_bunch:
                    peak_map[peak].add(chromatogram_node)
        edges = set()
        for peak, nodes in peak_map.items():
            for a, b in itertools.combinations(nodes, 2):
                edges.add(frozenset((a, b)))

        result = []
        for edge in edges:
            start, end = sorted(edge, key=lambda x: x.index)
            delta = -(start.neutral_mass - end.neutral_mass)
            chromatogram_edge = ChromatogramGraphEdge(
                start, end, UnknownTransition(delta), rt_error=start.center - end.center)
            result.append(chromatogram_edge)
            self.edges.add(chromatogram_edge)
        return result

    def build(self, query_width=2., transitions=None, **kwargs):
        for node in self.iterseeds():
            self.find_edges(node, query_width=query_width, transitions=transitions, **kwargs)

    def adjacency_matrix(self):
        adjmat = np.zeros((len(self), len(self)))
        nodes = self.nodes
        for node in nodes:
            for edge in node.edges:
                adjmat[edge.node_a.index, edge.node_b.index] = 1
        return adjmat

    def connected_components(self):
        pool = set(self.nodes)
        components = []

        i = 0
        while pool:
            i += 1
            current_component = set()
            visited = set()
            node = pool.pop()
            current_component.add(node)
            j = 0
            while current_component:
                j += 1
                node = current_component.pop()
                visited.add(node)
                for edge in node.edges:
                    if edge.node_a not in visited and edge.node_a in pool:
                        current_component.add(edge.node_a)
                        pool.remove(edge.node_a)
                    if edge.node_b not in visited and edge.node_b in pool:
                        current_component.add(edge.node_b)
                        pool.remove(edge.node_b)
            components.append(list(visited))
        return components
