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


cdef struct glycan_composition_vector:
    int* counts
    size_t size


cdef int initialize_glycan_composition_vector(size_t size, glycan_composition_vector* self) nogil
cdef int destroy_glycan_composition_vector(glycan_composition_vector* self) nogil
cdef int copy_glycan_composition_vector(glycan_composition_vector* self, glycan_composition_vector* into) nogil

cdef double glycan_composition_vector_distance(glycan_composition_vector* self, glycan_composition_vector* other) nogil
cdef int glycan_composition_vector_difference(glycan_composition_vector* self, glycan_composition_vector* other, glycan_composition_vector* into) nogil
cdef int glycan_composition_vector_addition(glycan_composition_vector* self, glycan_composition_vector* other, glycan_composition_vector* into) nogil



cdef class GlycanCompositionVectorContext:
    cdef:
        public list components
        public size_t component_count

    cdef glycan_composition_vector* encode_raw(self, glycan_composition)
    cpdef GlycanCompositionVector encode(self, glycan_composition)
    cpdef decode(self, GlycanCompositionVector gcvec)


cdef class GlycanCompositionVector:
    cdef:
        glycan_composition_vector* ptr

    @staticmethod
    cdef GlycanCompositionVector _create(glycan_composition_vector* ptr)

    cpdef double distance(self, GlycanCompositionVector other) except -1

    cpdef GlycanCompositionVector difference(self, GlycanCompositionVector other)
    cpdef GlycanCompositionVector addition(self, GlycanCompositionVector other)

    cpdef GlycanCompositionVector clone(self)


cdef class CachingGlycanCompositionVectorContext(GlycanCompositionVectorContext):
    cdef:
        public dict encode_cache
        public dict decode_cache