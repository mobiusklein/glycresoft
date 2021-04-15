cimport cython

from glypy.composition.ccomposition cimport CComposition


cdef dict mass_shift_index = dict()


cdef class MassShiftBase(object):

    cdef:
        public CComposition composition
        public CComposition tandem_composition
        public basestring name
        public long _hash
        public double mass
        public double tandem_mass


cdef class MassShift(MassShiftBase):
    pass


cdef class CompoundMassShift(MassShiftBase):

    cdef:
        public object counts



cdef MassShiftBase Unmodified
