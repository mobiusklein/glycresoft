cimport numpy as np

cpdef bint is_diagonal(np.ndarray[double, ndim=2] mat)

cdef class GlycanPriorRecord(object):
    cdef:
        public double score
        public bint matched
