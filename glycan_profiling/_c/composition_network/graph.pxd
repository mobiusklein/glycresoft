cdef class CompositionGraphNode(object):
    cdef:
        public object composition
        public ssize_t index
        public str _str
        public EdgeSet edges
        public Py_hash_t _hash
        public double _score
        public double internal_score
        public bint marked

cdef class EdgeSet(object):
    cdef:
        public dict store

    cpdef bint add_if_shorter(self, CompositionGraphEdge edge)
    cdef list get_edges(self)


cdef class CompositionGraphEdge(object):
    cdef:
        public CompositionGraphNode node1
        public CompositionGraphNode node2
        public long order
        public double weight
        public Py_hash_t _hash
        public str _str


    @staticmethod
    cdef CompositionGraphEdge _create(CompositionGraphNode node1, CompositionGraphNode node2, long order, double weight)

    cdef void _init(self)
    cpdef copy_for(self, CompositionGraphNode node1, CompositionGraphNode node2)



cdef class CompositionGraphBase(object):
    cdef:
        public list nodes
        public dict node_map
        public object _composition_normalizer
        public object distance_fn
        public EdgeSet edges
        public object neighborhoods

    cdef list get_edges(self)
    cpdef CompositionGraphBase copy(self)

cpdef reindex_graph(self)