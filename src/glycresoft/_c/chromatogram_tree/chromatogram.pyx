cimport cython
from cpython.list cimport PyList_Append, PyList_Size, PyList_GetItem

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak
from glypy.composition.ccomposition cimport CComposition
from glycresoft._c.chromatogram_tree.mass_shift cimport MassShiftBase, MassShift

from glypy.utils import uid

cdef MassShiftBase Unmodified = MassShift("Unmodified", CComposition())


cdef class ChromatogramTreeNode(object):
    def __init__(self, retention_time=None, scan_id=None, children=None, members=None,
                 node_type=Unmodified):
        if children is None:
            children = []
        if members is None:
            members = []
        self.retention_time = retention_time
        self.scan_id = scan_id
        self.children = children
        self.members = members
        self.node_type = node_type
        self._most_abundant_member = None
        self._neutral_mass = 0
        self._charge_states = set()
        self._recalculate()
        self._has_msms = None
        self.node_id = uid()

    def __reduce__(self):
        return self.__class__, (
            self.retention_time, self.scan_id, [c for c in self.children],
            list(self.members), self.node_type), self.__getstate__()

    def __getstate__(self):
        return {
            "node_id": self.node_id
        }

    def __setstate__(self, state):
        self.node_id = state['node_id']

    cpdef ChromatogramTreeNode clone(self):
        node = ChromatogramTreeNode(
            self.retention_time, self.scan_id, [c.clone() for c in self.children],
            list(self.members), node_type=self.node_type)
        node.node_id = self.node_id
        return node

    def _unspool_strip_children(self):
        node = ChromatogramTreeNode(
            self.retention_time, self.scan_id, [], list(self.members), node_type=self.node_type)
        yield node
        for child in self.children:
            for node in child._unspool_strip_children():
                yield node

    cpdef _recalculate(self):
        self._calculate_most_abundant_member()
        self._neutral_mass = self._most_abundant_member.neutral_mass
        self._charge_states = None
        self._has_msms = None

    cpdef _calculate_most_abundant_member(self):
        cdef:
            size_t i, n
            DeconvolutedPeak peak, candidate
        n = PyList_Size(self.members)
        if n == 1:
            self._most_abundant_member = self.members[0]
        else:
            if n == 0:
                self._most_abundant_member = None
            else:
                candidate = <DeconvolutedPeak>PyList_GetItem(self.members, 0)
                for i in range(1, n):
                    peak = <DeconvolutedPeak>PyList_GetItem(self.members, i)
                    if peak.intensity > candidate.intensity:
                        candidate = peak
                self._most_abundant_member = candidate

    @property
    def neutral_mass(self):
        if self._neutral_mass == 0:
            if self._most_abundant_member is not None:
                self._neutral_mass = self.get_neutral_mass()
        return self._neutral_mass

    cdef double get_neutral_mass(self):
        if self._most_abundant_member is not None:
            return self._most_abundant_member.neutral_mass
        else:
            return 0.0

    cpdef ChromatogramTreeNode _find(self, MassShiftBase node_type=Unmodified):
        if self.node_type == node_type:
            return self
        else:
            for child in self.children:
                match = child._find(node_type)
                if match is not None:
                    return match

    cpdef ChromatogramTreeNode find(self, MassShiftBase node_type=Unmodified):
        match = self._find(node_type)
        if match is not None:
            return match
        else:
            raise KeyError(node_type)

    cdef set get_contained_charge_states(self):
        cdef:
            set charges
            DeconvolutedPeak peak
            size_t i, n
        charges = set()
        n = PyList_Size(self.members)
        for i in range(n):
            peak = <DeconvolutedPeak>PyList_GetItem(self.members, i)
            charges.add(peak.charge)
        return charges

    cdef set get_charge_states(self):
        cdef:
            set charges
            size_t i, n
            ChromatogramTreeNode node

        charges = self.get_contained_charge_states()
        n = PyList_Size(self.children)
        for i in range(n):
            node = <ChromatogramTreeNode>PyList_GetItem(self.children, i)
            charges.update(node.get_charge_states())
        return charges

    @property
    def _contained_charge_states(self):
        if self._charge_states is None:
            self._charge_states = self.get_contained_charge_states()
        return self._charge_states

    cpdef set charge_states(self):
        return self.get_charge_states()

    @property
    def has_msms(self):
        if self._has_msms is None:
            self._has_msms = self._has_any_peaks_with_msms()
        return self._has_msms

    cdef double _total_intensity_members(self):
        total = 0.
        for peak in self.members:
            total += peak.intensity
        return total

    cdef double _total_intensity_children(self):
        total = 0.
        for child in self.children:
            total += child.total_intensity()
        return total

    cpdef double max_intensity(self):
        return self._most_abundant_member.intensity

    cpdef double total_intensity(self):
        return self._total_intensity_children() + self._total_intensity_members()

    def __eq__(self, other):
        return self.members == other.members

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.uid)

    def _has_any_peaks_with_msms(self):
        for peak in self.members:
            if peak.chosen_for_msms:
                return True
        for child in self.children:
            if child._has_any_peaks_with_msms():
                return True
        return False

    @property
    def peaks(self):
        peaks = list(self.members)
        for child in self.children:
            peaks.extend(child.peaks)
        return peaks