cimport cython
# from cpython.list cimport PyList_Append, PyList_Size, PyList_GetItem

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak
from glypy.composition.ccomposition cimport CComposition
from glycresoft._c.chromatogram_tree.mass_shift cimport MassShiftBase, MassShift


cdef MassShiftBase Unmodified


cdef class ChromatogramTreeNode(object):

    cdef:
        public list children
        public list members
        public double retention_time
        public object scan_id
        public MassShiftBase node_type
        public DeconvolutedPeak _most_abundant_member
        public double _neutral_mass
        public set _charge_states
        public object _has_msms
        public object node_id

    cpdef ChromatogramTreeNode clone(self)
    cpdef _recalculate(self)

    cdef set get_charge_states(self)
    cdef set get_contained_charge_states(self)

    cpdef set charge_states(self)

    cdef double get_neutral_mass(self)

    cpdef _calculate_most_abundant_member(self)

    cpdef ChromatogramTreeNode _find(self, MassShiftBase node_type=*)
    cpdef ChromatogramTreeNode find(self, MassShiftBase node_type=*)

    cdef double _total_intensity_members(self)
    cdef double _total_intensity_children(self)

    cpdef double max_intensity(self)
    cpdef double total_intensity(self)

