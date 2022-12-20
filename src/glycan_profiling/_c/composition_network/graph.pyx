cimport cython

from libc.stdlib cimport malloc, free, abs as labs

from collections import deque

from cpython.object cimport PyObject
from cpython.list cimport PyList_Size, PyList_GetItem, PyList_GET_ITEM
from cpython.set cimport PySet_Discard, PySet_Add
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_DelItem, PyDict_Items, PyDict_Values
from cpython.float cimport PyFloat_AsDouble
from cpython.int cimport PyInt_AsLong

from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenMonosaccharideResidue


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

    cdef list get_edges(self):
        return PyDict_Values(self.store)



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


cdef double INF = float('inf')


cdef class CompositionGraphBase(object):
    cdef list get_edges(self):
        return self.edges.get_edges()

    cpdef CompositionGraphBase copy(CompositionGraphBase self):
        cdef:
            CompositionGraphNode n1, n2
            CompositionGraphEdge edge, e
            CompositionGraphBase graph
            list edges, nodes
            size_t n, i

        graph = self.__class__([], self.distance_fn)
        graph._composition_normalizer = self._composition_normalizer.copy()
        for node in self.nodes:
            graph.add_node(node.clone())

        nodes = graph.nodes
        edges = self.get_edges()

        n = PyList_Size(edges)
        for i in range(n):
            edge = <CompositionGraphEdge>PyList_GET_ITEM(edges, i)
            n1 = nodes[edge.node1.index]
            n2 = nodes[edge.node2.index]
            e = edge.copy_for(n1, n2)
            graph.edges.add(e)
        graph.neighborhoods.update(self.neighborhoods.copy())
        return graph


cdef class DijkstraPathFinder(object):

    cdef:
        public object graph
        public CompositionGraphNode start
        public CompositionGraphNode end
        public dict distance
        public dict unvisited_finite_distance
        public double limit

    def __init__(self, graph, start, end, limit=INF):
        self.graph = graph
        self.start = start
        self.end = end
        assert start._str is not None
        assert end._str is not None
        self.distance = dict()
        self.distance[start._str] = 0
        self.unvisited_finite_distance = dict()
        self.limit = limit

    cpdef set _build_initial_key_set(self):
        cdef:
            size_t i, n
            CompositionGraphNode node
            set result
            list nodes
        result = set()
        nodes = self.graph.nodes
        n = PyList_Size(nodes)
        for i in range(n):
            node = <CompositionGraphNode>PyList_GetItem(nodes, i)
            assert node._str is not None
            result.add(node._str)
        return result

    cpdef CompositionGraphNode find_smallest_unvisited(self, set unvisited):
        cdef:
            double smallest_distance, distance
            object smallest_node, key
            list iterable
            tuple entry
            PyObject* ptemp

        smallest_distance = INF
        smallest_node = None

        if not unvisited:
            raise ValueError("The unvisited set must not be empty")

        if not self.unvisited_finite_distance:
            for key in unvisited:
                ptemp = PyDict_GetItem(self.distance, key)
                if ptemp == NULL:
                    distance = INF
                if distance <= smallest_distance:
                    smallest_node = key
                    smallest_distance = distance
            if smallest_node is None:
                raise ValueError(f"Failed to select a destination from unvisited set of size {len(unvisited)} "
                                 f"with smallest distance {smallest_distance}:\n{unvisited}\nlooking for {self.start._str}->{self.end._str}")
        else:
            iterable = PyDict_Items(self.unvisited_finite_distance)
            for node, distance in iterable:
                if distance <= smallest_distance:
                    smallest_distance = distance
                    smallest_node = node
            if smallest_node is None:
                raise ValueError(f"Failed to select a destination from unvisited set of size {len(unvisited)} "
                                 f"with smallest distance {smallest_distance}:\n{unvisited}\nlooking for {self.start._str}->{self.end._str}")
        return self.graph[smallest_node]

    cpdef find_path(self):
        cdef:
            set unvisited
            CompositionGraphNode current_node, terminal
            EdgeSet edges
            CompositionGraphEdge edge
            PyObject* ptemp
            double path_length, terminal_distance
            long n
            int tmp

        unvisited = self._build_initial_key_set()

        visit_queue = deque([self.start])
        n = 1
        while self.end._str in unvisited:
            if n > 0:
                n -= 1
                current_node = visit_queue.popleft()
            else:
                current_node = self.find_smallest_unvisited(unvisited)

            tmp = PySet_Discard(unvisited, current_node._str)
            if tmp == -1:
                raise KeyError(current_node._str)
            elif tmp == 0:
                continue
            if PyDict_GetItem(self.unvisited_finite_distance, current_node._str) != NULL:
                PyDict_DelItem(self.unvisited_finite_distance, current_node._str)
            edges = current_node.edges
            for edge in edges:
                terminal = edge._traverse(current_node)
                if terminal._str not in unvisited:
                    continue
                ptemp = PyDict_GetItem(self.distance, current_node._str)
                if ptemp == NULL:
                    path_length = INF
                else:
                    path_length = PyFloat_AsDouble(<object>ptemp)
                path_length += edge.order
                ptemp = PyDict_GetItem(self.distance, terminal._str)
                if ptemp == NULL:
                    terminal_distance = INF
                else:
                    terminal_distance = PyFloat_AsDouble(<object>ptemp)

                if terminal_distance > path_length:
                    self.distance[terminal._str] = path_length
                    if terminal._str in unvisited and terminal_distance < self.limit:
                        self.unvisited_finite_distance[terminal._str] = path_length
                if terminal_distance < self.limit:
                    visit_queue.append(terminal)
                    n += 1

    def search(self):
        self.find_path()
        ptemp = PyDict_GetItem(self.distance, self.end._str)
        if ptemp == NULL:
            return INF
        else:
            return <object>(ptemp)


cdef int initialize_glycan_composition_vector(size_t size, glycan_composition_vector* self) nogil:
    self.size = size
    self.counts = <int*>malloc(sizeof(int) * size)
    if self.counts == NULL:
        return 1
    return 0


cdef double glycan_composition_vector_distance(glycan_composition_vector* self, glycan_composition_vector* other) nogil:
    cdef:
        size_t i, n
        double distance
    distance = 0
    n = self.size
    if other.size != n:
        return INF
    for i in range(n):
        distance += labs(self.counts[i] - other.counts[i])
    return distance


cdef int glycan_composition_vector_difference(glycan_composition_vector* self, glycan_composition_vector* other, glycan_composition_vector* into) nogil:
    cdef:
        size_t i, n

    n = self.size
    if other.size != n:
        return 1

    if into.counts == NULL:
        into.counts = <int*>malloc(sizeof(int) * n)
        into.size = n
        if into.counts == NULL:
            return 2

    for i in range(n):
        into.counts[i] = self.counts[i] - other.counts[i]
    return 0


cdef int glycan_composition_vector_addition(glycan_composition_vector* self, glycan_composition_vector* other, glycan_composition_vector* into) nogil:
    cdef:
        size_t i, n

    n = self.size
    if other.size != n:
        return 1

    if into.counts == NULL:
        into.counts = <int*>malloc(sizeof(int) * n)
        into.size = n
        if into.counts == NULL:
            return 2

    for i in range(n):
        into.counts[i] = self.counts[i] + other.counts[i]
    return 0


cdef int destroy_glycan_composition_vector(glycan_composition_vector* self) nogil:
    free(self.counts)
    return 0


cdef int copy_glycan_composition_vector(glycan_composition_vector* self, glycan_composition_vector* into) nogil:
    cdef:
        size_t i, n

    n = self.size
    if into.size != self.size:
        free(into.counts)
        into.counts = NULL

    if into.counts == NULL:
        into.counts = <int*>malloc(sizeof(int) * n)
        into.size = n
        if into.counts == NULL:
            return 1

    for i in range(n):
        into.counts[i] = self.counts[i]

    return 0


cdef class GlycanCompositionVectorContext:

    def __init__(self, components):
        self.components = list(components)
        self.component_count = len(self.components)

    def __reduce__(self):
        return self.__class__, (self.components, )

    cdef glycan_composition_vector* encode_raw(self, glycan_composition):
        cdef glycan_composition_vector* ptr = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
        if initialize_glycan_composition_vector(self.component_count, ptr) == 1:
            raise MemoryError()
        for i in range(self.component_count):
            ptr.counts[i] = glycan_composition[self.components[i]]
        return ptr

    cpdef GlycanCompositionVector encode(self, glycan_composition):
        return GlycanCompositionVector._create(self.encode_raw(glycan_composition))

    cpdef decode(self, GlycanCompositionVector gcvec):
        cdef:
            size_t i
            int value
        gc = HashableGlycanComposition()
        if self.component_count != gcvec.ptr.size:
            raise ValueError("Dimensionality does not match")
        for i in range(self.component_count):
            value = gcvec.ptr.counts[i]
            if value != 0:
                gc[self.components[i]] = value
        return gc


cpdef _reconstruct_GlycanCompositionVector(state):
    cdef:
        glycan_composition_vector* ptr
        size_t i, n
        list counts
    counts = state[0]
    n = len(counts)
    ptr = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
    initialize_glycan_composition_vector(n, ptr)
    for i in range(n):
        ptr.counts[i] = PyInt_AsLong(counts[i])
    return GlycanCompositionVector._create(ptr)



@cython.no_gc
@cython.final
@cython.freelist(2000)
cdef class GlycanCompositionVector:

    @staticmethod
    cdef GlycanCompositionVector _create(glycan_composition_vector* ptr):
        cdef GlycanCompositionVector self = GlycanCompositionVector.__new__(GlycanCompositionVector)
        self.ptr = ptr
        return self

    def __reduce__(self):
        return _reconstruct_GlycanCompositionVector, (list(self), )

    def __dealloc__(self):
        if self.ptr != NULL:
            destroy_glycan_composition_vector(self.ptr)
            free(self.ptr)
            self.ptr = NULL

    def __getitem__(self, int i):
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        if i >= self.ptr.size or i < 0:
            raise IndexError(i)
        return self.ptr.counts[i]

    def __setitem__(self, int i, int value):
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        if i >= self.ptr.size or i < 0:
            raise IndexError(i)
        self.ptr.counts[i] = value

    def __len__(self):
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        return self.ptr.size

    def __repr__(self):
        return f"{self.__class__.__name__}({[self[i] for i in range(len(self))]})"

    cpdef double distance(self, GlycanCompositionVector other) except -1:
        if other is None:
            return INF
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        return glycan_composition_vector_distance(self.ptr, other.ptr)

    def __eq__(self, GlycanCompositionVector other):
        if other is None:
            return False
        return self.distance(other) == 0

    def __ne__(self, GlycanCompositionVector other):
        return not self == other

    cpdef GlycanCompositionVector difference(self, GlycanCompositionVector other):
        cdef:
             glycan_composition_vector* new
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        if other is None:
            raise TypeError("Cannot compute difference from `None`")

        new = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
        initialize_glycan_composition_vector(self.ptr.size, new)
        if new == NULL:
            raise MemoryError()
        if glycan_composition_vector_difference(self.ptr, other.ptr, new) == 2:
            raise MemoryError()
        return GlycanCompositionVector._create(new)

    cpdef GlycanCompositionVector addition(self, GlycanCompositionVector other):
        cdef:
             glycan_composition_vector* new
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        if other is None:
            raise TypeError("Cannot compute addition from `None`")

        new = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
        initialize_glycan_composition_vector(self.ptr.size, new)
        if new == NULL:
            raise MemoryError()
        if glycan_composition_vector_addition(self.ptr, other.ptr, new) == 2:
            raise MemoryError()
        return GlycanCompositionVector._create(new)

    def __sub__(self, GlycanCompositionVector other):
        return self.difference(other)

    def __add__(self, GlycanCompositionVector other):
        return self.addition(other)

    def __neg__(self):
        cdef:
             glycan_composition_vector* new

        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        new = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
        if new == NULL:
            raise MemoryError()
        initialize_glycan_composition_vector(self.ptr.size, new)

        for i in range(self.ptr.size):
            new.counts[i] = -self.ptr.counts[i]
        return GlycanCompositionVector._create(new)

    def __hash__(self):
        cdef:
            size_t i, n
            Py_hash_t hash_val

        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        hash_val = 0
        n = self.ptr.size
        for i in range(n):
            hash_val = (hash_val ^ self.ptr.counts[i]) * 82520UL
        return hash_val

    cpdef GlycanCompositionVector clone(self):
        cdef:
            glycan_composition_vector* new
        if self.ptr == NULL:
            raise ValueError("Object in invalid state")
        new = <glycan_composition_vector*>malloc(sizeof(glycan_composition_vector))
        if new == NULL:
            raise MemoryError()
        new.counts = NULL
        new.size = 0
        if copy_glycan_composition_vector(self.ptr, new) != 0:
            raise MemoryError()
        return GlycanCompositionVector._create(new)


cdef class CachingGlycanCompositionVectorContext(GlycanCompositionVectorContext):

    def __init__(self, components):
        super().__init__(components)
        self.encode_cache = {}
        self.decode_cache = {}

    cpdef GlycanCompositionVector encode(self, gc):
        cdef:
            PyObject* ptemp
            GlycanCompositionVector value

        ptemp = PyDict_GetItem(self.encode_cache, gc)
        if ptemp != NULL:
            return <GlycanCompositionVector>ptemp
        value = GlycanCompositionVector._create(self.encode_raw(gc))
        PyDict_SetItem(self.encode_cache, gc, value)
        PyDict_SetItem(self.decode_cache, value, gc)
        return value

    cpdef decode(self, GlycanCompositionVector gcvec):
        cdef:
            PyObject* ptemp
        ptemp = PyDict_GetItem(self.decode_cache, gcvec)
        if ptemp != NULL:
            return <object>ptemp

        value = super(CachingGlycanCompositionVectorContext, self).decode(gcvec)
        self.encode_cache[value] = gcvec
        self.decode_cache[gcvec] = value
        return value
