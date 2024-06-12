from collections import defaultdict

from typing import Any, Dict, Generic, List, Tuple, TypeVar

import numpy as np

from ms_deisotope.data_source import ProcessedScan


T = TypeVar('T')


class MassWrapper(Generic[T]):
    obj: T
    mass: float

    def __init__(self, obj, mass):
        self.obj = obj
        self.mass = mass

    def __repr__(self):
        return "MassWrapper(%s, %f)" % (self.obj, self.mass)


class NodeBase(object):
    def __init__(self, index: int):
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


class EdgeSet(object):
    store: Dict[Tuple[NodeBase, NodeBase], 'GraphEdgeBase']

    def __init__(self, store=None):
        if store is None:
            store = dict()
        self.store = store

    def edge_to(self, node1: NodeBase, node2: NodeBase):
        if node2.index < node1.index:
            node1, node2 = node2, node1
        return self.store.get((node1, node2))

    def add(self, edge: 'GraphEdgeBase'):
        self.store[edge.node1, edge.node2] = edge

    def remove(self, edge: 'GraphEdgeBase'):
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

    node1: NodeBase
    node2: NodeBase
    indices: Tuple[int, int]

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

    def __getitem__(self, key: NodeBase):
        if key == self.node1:
            return self.node2
        elif key == self.node2:
            return self.node1
        else:
            raise KeyError(key)

    def is_incoming(self, node: NodeBase):
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

    def copy_for(self, node1: NodeBase, node2: NodeBase):
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


NodeT = TypeVar('NodeT', bound=NodeBase)

class GraphBase(Generic[NodeT]):
    edges: EdgeSet
    nodes: List[NodeT]

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
        A = np.zeros((N, N), dtype=np.uint8)
        for edge in self.edges:
            A[edge.indices] = 1
        return A

    def topological_sort(self, adjacency_matrix: np.ndarray = None) -> List[NodeT]:
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix()
        else:
            adjacency_matrix = adjacency_matrix.copy()
        waiting = set()
        for i in range(adjacency_matrix.shape[0]):
            # Check for incoming edges. If no incoming
            # edges, add to the waiting set
            if adjacency_matrix[:, i].sum() == 0:
                waiting.add(i)

        ordered = list()
        while waiting:
            ix = waiting.pop()
            node = self[ix]
            ordered.append(node)
            # Get the set of outgoing edge terminal indices
            outgoing_edges = adjacency_matrix[ix, :]
            indices_of_neighbors = np.nonzero(outgoing_edges)[0]
            # For each outgoing edge
            for neighbor_ix in indices_of_neighbors:
                # Remove the edge
                adjacency_matrix[ix, neighbor_ix] = 0
                # Test for incoming edges
                if (adjacency_matrix[:, neighbor_ix] == 0).all():
                    waiting.add(neighbor_ix)
        if adjacency_matrix.sum() > 0:
            raise ValueError("%d edges left over" % (adjacency_matrix.sum(),))
        else:
            return ordered

    def path_lengths(self):
        adjacency_matrix = self.adjacency_matrix()
        distances = np.zeros(len(self))
        for node in self.topological_sort():
            incoming_edges = np.nonzero(adjacency_matrix[:, node.index])[0]
            if incoming_edges.shape[0] == 0:
                distances[node.index] = 0
            else:
                longest_path_so_far = distances[incoming_edges].max()
                distances[node.index] = longest_path_so_far + 1
        return distances

    def connected_components(self) -> List[List[NodeT]]:
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


class ScanNode(NodeBase):
    scan: ProcessedScan

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
        return "ScanNode(%s)" % (self.scan.id,)


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


class ScanGraph(GraphBase[ScanNode]):
    scans: List[ProcessedScan]
    node_map: Dict[str, ScanNode]

    def __init__(self, scans):
        GraphBase.__init__(self)
        self.scans = scans
        self.nodes = [ScanNode(scan, i) for i, scan in enumerate(scans)]
        self.node_map = {node.scan.id: node for node in self.nodes}
        self.edges = AnnotatedMultiEdgeSet()

    def get_node_by_id(self, scan_id: str):
        return self.node_map[scan_id]

    def add_edge_between(self, scan_id1: str, scan_id2: str, annotation=None):
        node1 = self.get_node_by_id(scan_id1)
        node2 = self.get_node_by_id(scan_id2)
        edge = ScanGraphEdge(node1, node2, annotation)
        self.edges.add(edge)
