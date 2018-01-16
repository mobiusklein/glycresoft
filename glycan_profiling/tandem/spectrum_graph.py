from collections import defaultdict

import numpy as np
from glycopeptidepy.structure import sequence_composition


class MassWrapper(object):
    def __init__(self, obj, mass):
        self.obj = obj
        self.mass = mass

    def __repr__(self):
        return "MassWrapper(%s, %f)" % (self.obj, self.mass)


class NodeBase(object):
    def __init__(self, index):
        self.index = index
        self._hash = hash(self.index)
        self.edges = EdgeSet()

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.index == other.index

    def has_incoming_edge(self):
        for edge in self.edges:
            if edge.is_incoming(self):
                return True
        return False


class PeakNode(NodeBase):
    def __init__(self, peak):
        self.peak = peak
        NodeBase.__init__(self, peak.index.neutral_mass)
        self._hash = hash(peak)
        self.mass = self.peak.neutral_mass
        self.charge = self.peak.charge

    @property
    def intensity(self):
        return self.peak.intensity

    def __eq__(self, other):
        return self.peak == other.peak

    def __repr__(self):
        return "PeakNode({}, {}, {})".format(self.mass, self.charge, self.index)


class EdgeSet(object):

    def __init__(self, store=None):
        if store is None:
            store = dict()
        self.store = store

    def edge_to(self, node1, node2):
        if node2.index < node1.index:
            node1, node2 = node2, node1
        return self.store.get((node1, node2))

    def add(self, edge):
        self.store[edge.node1, edge.node2] = edge

    def remove(self, edge):
        self.store.pop((edge.node1, edge.node2))

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store.values())

    def __repr__(self):
        template = '{self.__class__.__name__}({self.store})'
        return template.format(self=self)


class AnnotatedMultiEdgeSet(EdgeSet):
    def __init__(self, store=None):
        if store is None:
            store = defaultdict(dict)
        EdgeSet.__init__(self, store)

    def __iter__(self):
        for subgroup in self.store.values():
            for edge in subgroup.values():
                yield edge

    def add(self, edge):
        self.store[edge.node1, edge.node2][edge.annotation] = edge

    def __len__(self):
        return sum(map(len, self.store.values()))

    def __repr__(self):
        template = '{self.__class__.__name__}({self.store})'
        return template.format(self=self)


class GraphEdgeBase(object):
    __slots__ = ["node1", "node2", "_hash", "_str", "indices"]

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

        self.indices = (self.node1.index, self.node2.index)

        self._hash = self._make_hash()
        self._str = self._make_str()

        node1.edges.add(self)
        node2.edges.add(self)

    def _make_hash(self):
        return hash((self.node1, self.node2))

    def _make_str(self):
        return "GraphEdge(%s)" % ', '.join(map(str, (self.node1, self.node2)))

    def __getitem__(self, key):
        if key == self.node1:
            return self.node2
        elif key == self.node2:
            return self.node1
        else:
            raise KeyError(key)

    def is_incoming(self, node):
        if node is self.node2:
            return True
        elif node is self.node1:
            return False
        else:
            return False

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
        return self.__class__, (self.node1, self.node2,)

    def copy_for(self, node1, node2):
        return self.__class__(node1, node2,)

    def remove(self):
        try:
            self.node1.edges.remove(self)
        except KeyError:
            pass
        try:
            self.node2.edges.remove(self)
        except KeyError:
            pass


class PeakGraphEdge(GraphEdgeBase):
    __slots__ = ["annotation", "weight"]

    def __init__(self, node1, node2, annotation, weight=1.0):
        self.annotation = annotation
        self.weight = weight

        GraphEdgeBase.__init__(self, node1, node2)
        # self.node1 = node1
        # self.node2 = node2
        # self.indices = (self.node1.index, self.node2.index)
        # self._hash = hash((node1, node2, annotation))
        # self._str = "PeakGraphEdge(%s)" % ', '.join(map(str, (node1, node2, annotation)))

        # node1.edges.add(self)
        # node2.edges.add(self)

    def _make_hash(self):
        return hash((self.node1, self.node2, self.annotation))

    def _make_str(self):
        return "PeakGraphEdge(%s)" % ', '.join(map(str, (self.node1, self.node2, self.annotation)))

    def __reduce__(self):
        return self.__class__, (self.node1, self.node2, self.annotation, self.weight)

    def copy_for(self, node1, node2):
        return self.__class__(node1, node2, self.annotation, self.weight)


class GraphBase(object):
    def __init__(self, edges=None):
        if edges is None:
            edges = EdgeSet()
        self.edges = edges
        self.nodes = []

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, i):
        node = self.nodes[i]
        return node

    def __len__(self):
        return len(self.nodes)

    def adjacency_matrix(self):
        N = len(self)
        A = np.zeros((N, N))
        for edge in self.edges:
            A[edge.indices] = 1
        return A

    def topological_sort(self, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix()
        else:
            adjacency_matrix = adjacency_matrix.copy()
        waiting = set()
        for i in range(adjacency_matrix.shape[0]):
            # Check for incoming edges. If no incoming
            # edges, add to the waiting set
            if adjacency_matrix[:, i].sum() == 0:
                waiting.add(self[i])

        ordered = list()
        while waiting:
            node = waiting.pop()
            ordered.append(node)
            # Get the set of outgoing edge terminal indices
            outgoing_edges = adjacency_matrix[node.index, :]
            indices_of_neighbors = np.nonzero(outgoing_edges)[0]
            # For each outgoing edge
            for neighbor_ix in indices_of_neighbors:
                # Remove the edge
                adjacency_matrix[node.index, neighbor_ix] = 0
                # Test for incoming edges
                if (adjacency_matrix[:, neighbor_ix] == 0).all():
                    waiting.add(self[neighbor_ix])
        if adjacency_matrix.sum() > 0:
            raise ValueError("%d edges left over" % (adjacency_matrix.sum(),))
        else:
            return ordered

    def path_lengths(self):
        adjacency_matrix = self.adjacency_matrix()
        distances = np.zeros(len(self))
        for node in self.topological_sort():
            longest_path_so_far = adjacency_matrix[:, node.index].max()
            distances[node.index] = longest_path_so_far + 1
        return distances

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
                    if edge.node1 not in visited and edge.node1 in pool:
                        current_component.add(edge.node1)
                        pool.remove(edge.node1)
                    if edge.node2 not in visited and edge.node2 in pool:
                        current_component.add(edge.node2)
                        pool.remove(edge.node2)
            components.append(list(visited))
        return components


class PeakGraph(GraphBase):
    def __init__(self, peak_set):
        GraphBase.__init__(self)
        self.peak_set = peak_set.clone()
        self.nodes = [PeakNode(p) for p in self.peak_set]

    def add_edges(self, blocks, error_tolerance=1e-5):
        for peak in self.peak_set:
            start_node = self.nodes[peak.index.neutral_mass]
            for block in blocks:
                delta = peak.neutral_mass + block.mass
                matched_peaks = self.peak_set.all_peaks_for(delta, error_tolerance)
                if matched_peaks:
                    for end_peak in matched_peaks:
                        end_node = self.nodes[end_peak.index.neutral_mass]
                        edge = PeakGraphEdge(start_node, end_node, block)
                        self.edges.add(edge)


class ScanNode(NodeBase):
    def __init__(self, scan, index):
        self.scan = scan
        NodeBase.__init__(self, index)
        self.edges = AnnotatedMultiEdgeSet()

    @property
    def id(self):
        return self.scan.id

    @property
    def precursor_information(self):
        return self.scan.precursor_information

    def __repr__(self):
        return "ScanNode(%s)" % self.scan.id


class ScanGraphEdge(GraphEdgeBase):
    __slots__ = ["annotation"]

    def __init__(self, node1, node2, annotation):
        self.annotation = annotation
        GraphEdgeBase.__init__(self, node1, node2)

    def _make_hash(self):
        return hash((self.node1, self.node2, self.annotation))

    def _make_str(self):
        return "ScanGraphEdge(%s)" % ', '.join(map(str, (self.node1, self.node2, self.annotation)))

    def __reduce__(self):
        return self.__class__, (self.node1, self.node2, self.annotation)

    def copy_for(self, node1, node2):
        return self.__class__(node1, node2, self.annotation)


class ScanGraph(GraphBase):
    def __init__(self, scans):
        GraphBase.__init__(self)
        self.scans = scans
        self.nodes = [ScanNode(scan, i) for i, scan in enumerate(scans)]
        self.node_map = {node.scan.id: node for node in self.nodes}
        self.edges = AnnotatedMultiEdgeSet()

    def get_node_by_id(self, scan_id):
        return self.node_map[scan_id]

    def add_edge_between(self, scan_id1, scan_id2, annotation=None):
        node1 = self.get_node_by_id(scan_id1)
        node2 = self.get_node_by_id(scan_id2)
        edge = ScanGraphEdge(node1, node2, annotation)
        self.edges.add(edge)
