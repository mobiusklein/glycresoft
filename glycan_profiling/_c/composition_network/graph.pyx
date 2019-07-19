cimport cython

from cpython.object cimport PyObject
from cpython.list cimport PyList_Size, PyList_GetItem
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem


@cython.freelist(5000)
cdef class CompositionGraphNode(object):

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
        cdef:
            CompositionGraphNode other_typed
        if isinstance(self, CompositionGraphNode):
            if isinstance(other, CompositionGraphNode):
                other_typed = <CompositionGraphNode>other
                return self._str == other_typed._str
            else:
                return self._str == str(other)
        else:
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


cdef class EdgeSet(object):

    def __init__(self, store=None):
        if store is None:
            store = dict()
        self.store = dict(store)

    def edge_to(self, CompositionGraphNode node1, CompositionGraphNode node2):
        if node2.index < node1.index:
            node1, node2 = node2, node1
        return self.store[node1, node2]

    def add(self, CompositionGraphEdge edge):
        self.store[edge.node1, edge.node2] = edge

    cpdef bint add_if_shorter(self, CompositionGraphEdge edge):
        cdef:
            object key
            PyObject* prev_ptr
            CompositionGraphEdge prev

        key = (edge.node1, edge.node2)
        prev_ptr = PyDict_GetItem(self.store, key)
        if prev_ptr == NULL:
            PyDict_SetItem(self.store, key, edge)
            return True
        else:
            prev = <CompositionGraphEdge>prev_ptr
            if prev.order < edge.order:
                return False
            else:
                PyDict_SetItem(self.store, key, edge)
                return True

    def remove(self, CompositionGraphEdge edge):
        self.store.pop((edge.node1, edge.node2))

    def __iter__(self):
        return iter(self.store.values())

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return str(set(self.store.values()))

    def __eq__(self, other):
        return self.store == other.store


@cython.freelist(15000)
cdef class CompositionGraphEdge(object):

    def __init__(self, node1, node2, order, weight=1.0):
        self.node1 = node1
        self.node2 = node2
        self.order = order if order > 0 else 1
        self.weight = weight
        self._init()

    @staticmethod
    cdef CompositionGraphEdge _create(CompositionGraphNode node1, CompositionGraphNode node2, long order, double weight):
        cdef:
            CompositionGraphEdge self

        self = CompositionGraphEdge.__new__(CompositionGraphEdge)
        self.node1 = node1
        self.node2 = node2
        self.order = order
        self.weight = weight
        self._init()
        return self

    cdef void _init(self):
        cdef:
            tuple key
        key = (self.node1, self.node2, self.order)
        self._hash = hash(key)
        self._str = "(%s, %s, %s)" % key
        self.node1.edges.add_if_shorter(self)
        self.node2.edges.add_if_shorter(self)

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

    cpdef copy_for(self, CompositionGraphNode node1, CompositionGraphNode node2):
        return CompositionGraphEdge._create(node1, node2, self.order, self.weight)

    def remove(self):
        try:
            self.node1.edges.remove(self)
        except KeyError:
            pass
        try:
            self.node2.edges.remove(self)
        except KeyError:
            pass


@cython.binding(True)
cpdef reindex_graph(self):
    cdef:
        size_t i, n
        CompositionGraphNode node
        list nodes

    nodes = self.nodes
    n = PyList_Size(nodes)
    for i in range(n):
        node = <CompositionGraphNode>PyList_GetItem(nodes, i)
        node.index = i
