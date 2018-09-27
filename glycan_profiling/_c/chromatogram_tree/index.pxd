cimport cython


cdef class MassObject(object):
    cdef:
        public object obj
        public double mass


cpdef (ssize_t, bint) binary_search_with_flag(list array, double mass, double error_tolerance=*)


cdef class ChromatogramIndex(object):
    cdef:
        list _chromatograms

    cpdef (ssize_t, bint) _binary_search(self, double mass, double error_tolerance=*)
    cpdef object find_all_by_mass(self, double mass, double ppm_error_tolerance=*)
    cpdef object find_mass(self, double mass, double ppm_error_tolerance=*)
    cpdef object mass_between(self, double low, double high)

    cdef inline MassObject get_index(self, size_t i)
    cdef inline size_t get_size(self)


