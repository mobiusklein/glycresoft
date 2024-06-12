cimport cython

from ms_deisotope._c.peak_set cimport DeconvolutedPeak
from glycopeptidepy._c.structure.fragment cimport FragmentBase

ctypedef fused FragmentType:
    FragmentBase
    object


@cython.final
@cython.freelist(1000000)
cdef class PeakFragmentPair(object):

    cdef:
        public DeconvolutedPeak peak
        public object fragment
        public str fragment_name
        public Py_hash_t _hash

    @staticmethod
    cdef PeakFragmentPair _create(DeconvolutedPeak peak, fragment)
    @staticmethod
    cdef PeakFragmentPair _create_simple(DeconvolutedPeak peak, FragmentBase fragment)

    cpdef double mass_accuracy(self)


@cython.final
@cython.freelist(1000000)
cdef class PeakPairTransition(object):
    cdef:
        public DeconvolutedPeak start
        public DeconvolutedPeak end
        public object annotation
        public tuple key
        public Py_hash_t _hash

    @staticmethod
    cdef PeakPairTransition _create(DeconvolutedPeak start, DeconvolutedPeak end, object annotation)


cdef class SpectrumGraph(object):
    cdef:
        public set transitions
        public object by_first
        public object by_second

    cpdef add(self, DeconvolutedPeak p1, DeconvolutedPeak p2, object annotation)
    cpdef list topological_sort(self, adjacency_matrix=*)

    cpdef list paths_starting_at(self, ix, int limit=*)
    cpdef list paths_ending_at(self, ix, int limit=*)

    cpdef list transitive_closure(self, list paths, int limit=*)


# forward declaration
cdef class FragmentMatchMap(object)


@cython.freelist(10000)
cdef class _FragmentIndexBase(object):
    cdef:
        public FragmentMatchMap fragment_set
        public object _mapping

    cdef int _create_mapping(self)
    cdef void invalidate(self)
    cdef list getitem(self, object key)
    cdef bint has_key(self, object key)


cdef class ByFragmentIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByFragmentIndex _create(FragmentMatchMap fragment_set)


cdef class ByPeakIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByPeakIndex _create(FragmentMatchMap fragment_set)


cdef class ByPeakIndexIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByPeakIndexIndex _create(FragmentMatchMap fragment_set)


@cython.final
@cython.freelist(1000000)
cdef class FragmentMatchMap(object):
    cdef:
        public set members
        public ByFragmentIndex by_fragment
        public ByPeakIndex by_peak
        public ByPeakIndexIndex by_peak_index

    cpdef add(self, peak, fragment=*)
    cdef void _add_direct(self, PeakFragmentPair pair)

    cpdef set fragments(self)
    cpdef FragmentMatchMap copy(self)
    cpdef FragmentMatchMap clone(self)
    cpdef clear(self)

    cpdef list peaks_for(self, fragment)
    cpdef list fragments_for(self, peak)

    cpdef invalidate(self)
    cpdef set peak_indices(self)