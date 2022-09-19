
cdef class MassObject(object):
    cdef:
        public object obj
        public double mass


cdef class NeutralMassDatabaseImpl(object):
    cdef:
        public list structures
        public object mass_getter

    cdef list _prepare(self, list structures, bint sort=*)
    cpdef size_t search_binary(self, double mass, double error_tolerance=*)

    cdef size_t get_size(self)
    cdef object get_item(self, size_t i)
    cpdef list search_mass(self, double mass, double error_tolerance=*)
    cpdef object search_between(self, double lower, double higher)