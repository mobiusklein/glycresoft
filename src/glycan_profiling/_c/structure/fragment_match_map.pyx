# cython: embedsignature=True
from collections import defaultdict

cimport cython
from cpython cimport PyObject
from cpython cimport PyList_Append, PyList_Size, PyList_GetItem
from cpython cimport PySet_Add
from cpython cimport PyDict_GetItem, PyDict_SetItem, PyDict_Contains
from cpython cimport PyInt_AsLong

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak

from glycopeptidepy._c.structure.fragment cimport FragmentBase, SimpleFragment, PeptideFragment


np.import_array()

@cython.final
@cython.freelist(1000000)
cdef class PeakFragmentPair(object):

    @staticmethod
    cdef PeakFragmentPair _create(DeconvolutedPeak peak, fragment):
        cdef PeakFragmentPair self = PeakFragmentPair.__new__(PeakFragmentPair)
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.name
        self._hash = hash(peak.mz)
        return self

    @staticmethod
    cdef PeakFragmentPair _create_simple(DeconvolutedPeak peak, FragmentBase fragment):
        cdef PeakFragmentPair self = PeakFragmentPair.__new__(PeakFragmentPair)
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.get_name()
        self._hash = hash(peak.mz)
        return self

    def __init__(self, peak, fragment):
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.name
        self._hash = hash(self.peak.mz)

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(self, PeakFragmentPair):
            return False
        return (self.peak == (<PeakFragmentPair>other).peak) and (
            self.fragment_name == (<PeakFragmentPair>other).fragment_name)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return self._hash

    def __reduce__(self):
        return self.__class__, (self.peak, self.fragment)

    def clone(self):
        return self.__class__(self.peak, self.fragment)

    def __repr__(self):
        return "PeakFragmentPair(%r, %r)" % (self.peak, self.fragment)

    def __iter__(self):
        yield self.peak
        yield self.fragment

    @cython.cdivision(True)
    cpdef double mass_accuracy(self):
        return (self.peak.neutral_mass - self.fragment.mass) / self.fragment.mass


@cython.final
@cython.freelist(1000000)
cdef class PeakPairTransition(object):

    @staticmethod
    cdef PeakPairTransition _create(DeconvolutedPeak start, DeconvolutedPeak end, object annotation):
        cdef PeakPairTransition self = PeakPairTransition.__new__(PeakPairTransition)
        self.start = start
        self.end = end
        self.annotation = annotation
        # The indices of the start peak and end peak
        self.key = (self.start.index.neutral_mass, self.end.index.neutral_mass)
        self._hash = hash(self.key)
        return self

    def __init__(self, start, end, annotation):
        self.start = start
        self.end = end
        self.annotation = annotation
        # The indices of the start peak and end peak
        self.key = (self.start.index.neutral_mass, self.end.index.neutral_mass)
        self._hash = hash(self.key)

    def __reduce__(self):
        return self.__class__, (self.start, self.end, self.annotation)

    def __eq__(self, other):
        cdef:
            PeakPairTransition typed_other
        if other is None:
            return False
        else:
            typed_other = <PeakPairTransition?>other
        # if self.key != typed_other.key:
        #     return False
        if not self.start._eq(typed_other.start):
            return False
        elif not self.end._eq(typed_other.end):
            return False
        elif self.annotation != typed_other.annotation:
            return False
        return True

    def __iter__(self):
        yield self.start
        yield self.end
        yield self.annotation

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return ("{self.__class__.__name__}({self.start.neutral_mass:0.3f} "
                "-{self.annotation}-> {self.end.neutral_mass:0.3f})").format(self=self)


cdef class SpectrumGraph(object):
    def __init__(self):
        self.transitions = set()
        self.by_first = defaultdict(list)
        self.by_second = defaultdict(list)

    cpdef add(self, DeconvolutedPeak p1, DeconvolutedPeak p2, object annotation):
        if p1._index.neutral_mass > p2._index.neutral_mass:
            temp = p2
            p2 = p1
            p1 = temp
        trans = PeakPairTransition._create(p1, p2, annotation)
        PySet_Add(self.transitions, trans)
        self.by_first[trans.key[0]].append(trans)
        self.by_second[trans.key[1]].append(trans)

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        return "{self.__class__.__name__}({size})".format(
            self=self, size=len(self.transitions))

    def _get_maximum_index(self):
        try:
            trans = max(self.transitions, key=lambda x: x.key[1])
            return trans.key[1]
        except ValueError:
            return 0

    def adjacency_matrix(self):
        n = self._get_maximum_index() + 1
        A = np.zeros((n, n))
        for trans in self.transitions:
            A[trans.key] = 1
        return A

    cpdef list topological_sort(self, adjacency_matrix=None):
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency_matrix()
        else:
            adjacency_matrix = adjacency_matrix.copy()

        waiting = set()
        for i in range(adjacency_matrix.shape[0]):
            # Check for incoming edges. If no incoming
            # edges, add to the waiting set
            if adjacency_matrix[:, i].sum() == 0:
                waiting.add(i)
        ordered = list()
        while waiting:
            ix = waiting.pop()
            ordered.append(ix)
            outgoing_edges = adjacency_matrix[ix, :]
            indices_of_neighbors = np.nonzero(outgoing_edges)[0]
            # For each outgoing edge
            for neighbor_ix in indices_of_neighbors:
                # Remove the edge
                adjacency_matrix[ix, neighbor_ix] = 0
                # Test for incoming edges
                if (adjacency_matrix[:, neighbor_ix] == 0).all():
                    waiting.add(neighbor_ix)
        if adjacency_matrix.sum() > 0:
            raise ValueError("%d edges left over" % (adjacency_matrix.sum(),))
        else:
            return ordered

    def path_lengths(self):
        adjacency_matrix = self.adjacency_matrix()
        distances = np.zeros(self._get_maximum_index() + 1)
        for ix in self.topological_sort():
            incoming_edges = np.nonzero(adjacency_matrix[:, ix])[0]
            if incoming_edges.shape[0] == 0:
                distances[ix] = 0
            else:
                longest_path_so_far = distances[incoming_edges].max()
                distances[ix] = longest_path_so_far + 1
        return distances

    cpdef list paths_starting_at(self, ix, int limit=-1):
        paths = []
        for trans in self.by_first[ix]:
            paths.append([trans])
        finished_paths = []
        while True:
            extended_paths = []
            for path in paths:
                terminal = path[-1]
                edges = self.by_first[terminal.key[1]]
                if not edges:
                    finished_paths.append(path)
                for trans in edges:
                    extended_paths.append(path + [trans])
            paths = extended_paths
            if len(paths) == 0:
                break
        return self.transitive_closure(finished_paths, limit)

    cpdef list paths_ending_at(self, ix, int limit=-1):
        paths = []
        for trans in self.by_second[ix]:
            paths.append([trans])
        finished_paths = []
        while True:
            extended_paths = []
            for path in paths:
                terminal = path[0]
                edges = self.by_second[terminal.key[0]]
                if not edges:
                    finished_paths.append(path)
                for trans in edges:
                    new_path = [trans]
                    new_path.extend(path)
                    extended_paths.append(new_path)
            paths = extended_paths
            if len(paths) == 0:
                break
        return self.transitive_closure(finished_paths, limit)

    cpdef list transitive_closure(self, list paths, int limit=-1):
        cdef:
            list path, index_list
            list node_sets, keep, node_sets_items
            set node_set, other_node_set
            dict by_node
            PyObject* tmp
            object conv
            PeakPairTransition node
            size_t i, j, n, m, k, q, v
            bint is_enclosed
        paths = sorted(paths, key=len, reverse=True)
        # precompute node_sets for each path
        node_sets = []
        by_node = {}
        for i in range(PyList_Size(paths)):
            path = <list>PyList_GetItem(paths, i)
            # track all nodes by index in this path in node_set
            node_set = set()
            for j in range(PyList_Size(path)):
                node = <PeakPairTransition>PyList_GetItem(path, j)
                # add the start node index and end node index to the
                # set of nodes on this path
                # node_set.update(node.key)
                if j == 0:
                    conv = node.start._index.neutral_mass
                    node_set.add(conv)
                    tmp = PyDict_GetItem(by_node, conv)
                    if tmp == NULL:
                        index_list = []
                        PyDict_SetItem(by_node, conv, index_list)
                    else:
                        index_list = <list>tmp
                    index_list.append(i)
                conv = node.end._index.neutral_mass
                node_set.add(conv)
                tmp = PyDict_GetItem(by_node, conv)
                if tmp == NULL:
                    index_list = []
                    PyDict_SetItem(by_node, conv, index_list)
                else:
                    index_list = <list>tmp
                index_list.append(i)
            node_sets.append(node_set)

        keep = []
        k = 0
        for i in range(PyList_Size(paths)):
            path = <list>PyList_GetItem(paths, i)
            is_enclosed = False
            node_set = <set>PyList_GetItem(node_sets, i)
            n = len(node_set)
            node = <PeakPairTransition>PyList_GetItem(path, 0)
            conv = node.start._index.neutral_mass
            index_list = <list>PyDict_GetItem(by_node, conv)
            for q in range(PyList_Size(index_list)):
                j = PyInt_AsLong(<object>PyList_GetItem(index_list, q))
                if i == j:
                    continue
                other_node_set = <set>PyList_GetItem(node_sets, j)
                m = len(other_node_set)
                if m < n:
                    break
                if node_set < other_node_set:
                    is_enclosed = True
                    break
            if not is_enclosed:
                keep.append(path)
                k += 1
            if limit > 0 and limit == k:
                break
        return keep

    def longest_paths(self, int limit=-1):
        cdef:
            list paths, segment
        # get all distinct paths
        paths = []
        for ix in np.argsort(self.path_lengths())[::-1]:
            segment = self.paths_ending_at(ix, limit)
            paths.extend(segment)
            if PyList_Size(segment) == 0:
                break
            # remove redundant paths
            paths = self.transitive_closure(paths, limit)
            if limit > 0:
                if PyList_Size(paths) > limit:
                    break
        return paths


@cython.freelist(10000)
cdef class _FragmentIndexBase(object):
    def __init__(self, fragment_set):
        self.fragment_set = fragment_set
        self._mapping = None

    cdef int _create_mapping(self):
        return 0

    @property
    def mapping(self):
        if self._mapping is None:
            self._create_mapping()
        return self._mapping

    cdef void invalidate(self):
        self._mapping = None

    cdef list getitem(self, object key):
        cdef:
            PyObject* ptemp
        if self._mapping is None:
            self._create_mapping()
        ptemp = PyDict_GetItem(self._mapping, key)
        if ptemp == NULL:
            return []
        else:
            return <list>ptemp

    cdef bint has_key(self, object key):
        if self._mapping is None:
            self._create_mapping()
        return PyDict_Contains(self._mapping, key)

    def __getitem__(self, key):
        return self.getitem(key)

    def __iter__(self):
        return iter(self.mapping)

    def items(self):
        return self.mapping.items()

    def keys(self):
        return self.mapping.keys()

    def values(self):
        return self.mapping.values()

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return self.has_key(key)

    def __str__(self):
        return str(self.mapping)


cdef class ByFragmentIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByFragmentIndex _create(FragmentMatchMap fragment_set):
        cdef ByFragmentIndex self = ByFragmentIndex.__new__(ByFragmentIndex)
        self.fragment_set = fragment_set
        self._mapping = None
        return self

    cdef int _create_mapping(self):
        cdef:
            object obj
            PyObject* ptemp
            PeakFragmentPair pair
            list bucket
        self._mapping = {}
        for obj in self.fragment_set.members:
            pair = <PeakFragmentPair>obj
            ptemp = PyDict_GetItem(self._mapping, pair.fragment)
            if ptemp == NULL:
                bucket = [pair.peak]
                PyDict_SetItem(self._mapping, pair.fragment, bucket)
            else:
                bucket = <list>ptemp
                bucket.append(pair.peak)
        return 0


cdef class ByPeakIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByPeakIndex _create(FragmentMatchMap fragment_set):
        cdef ByPeakIndex self = ByPeakIndex.__new__(ByPeakIndex)
        self.fragment_set = fragment_set
        self._mapping = None
        return self

    cdef int _create_mapping(self):
        cdef:
            object obj
            PyObject* ptemp
            PeakFragmentPair pair
            list bucket
        self._mapping = {}
        for obj in self.fragment_set.members:
            pair = <PeakFragmentPair>obj
            ptemp = PyDict_GetItem(self._mapping, pair.peak)
            if ptemp == NULL:
                bucket = [pair.fragment]
                PyDict_SetItem(self._mapping, pair.peak, bucket)
            else:
                bucket = <list>ptemp
                bucket.append(pair.fragment)
        return 0


cdef class ByPeakIndexIndex(_FragmentIndexBase):

    @staticmethod
    cdef ByPeakIndexIndex _create(FragmentMatchMap fragment_set):
        cdef ByPeakIndexIndex self = ByPeakIndexIndex.__new__(ByPeakIndexIndex)
        self.fragment_set = fragment_set
        self._mapping = None
        return self

    cdef int _create_mapping(self):
        cdef:
            object obj
            PyObject* ptemp
            PeakFragmentPair pair
            DeconvolutedPeak peak
            list bucket
            dict mapping
        mapping = self._mapping = {}
        for obj in self.fragment_set.members:
            pair = <PeakFragmentPair>obj
            peak = pair.peak
            ptemp = PyDict_GetItem(mapping, peak._index.neutral_mass)
            if ptemp == NULL:
                bucket = [pair.fragment]
                PyDict_SetItem(mapping, peak._index.neutral_mass, bucket)
            else:
                bucket = <list>ptemp
                PyList_Append(bucket, pair.fragment)
        return 0



@cython.final
@cython.freelist(1000000)
cdef class FragmentMatchMap(object):
    def __init__(self):
        self.members = set()
        self.by_fragment = ByFragmentIndex._create(self)
        self.by_peak = ByPeakIndex._create(self)
        self.by_peak_index = ByPeakIndexIndex._create(self)

    cpdef add(self, peak, fragment=None):
        if fragment is not None:
            if isinstance(fragment, FragmentBase):
                peak = PeakFragmentPair._create_simple(<DeconvolutedPeak>peak, <FragmentBase>fragment)
            else:
                peak = PeakFragmentPair._create(<DeconvolutedPeak>peak, fragment)
        PySet_Add(self.members, peak)

    cdef void _add_direct(self, PeakFragmentPair pair):
        PySet_Add(self.members, pair)

    def pairs_by_name(self, name):
        pairs = []
        for pair in self.members:
            if pair.fragment_name == name:
                pairs.append(pair)
        return pairs

    cpdef list fragments_for(self, peak):
        return self.by_peak.getitem(peak)

    cpdef list peaks_for(self, fragment):
        return self.by_fragment.getitem(fragment)

    def __eq__(self, other):
        return self.members == other.members

    def __ne__(self, other):
        return self.members != other.members

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def items(self):
        for peak, fragment in self.members:
            yield fragment, peak

    def values(self):
        for pair in self.members:
            yield pair.peak

    cpdef set fragments(self):
        cdef:
            set frags
            object obj
            PeakFragmentPair pair
        frags = set()
        for obj in self.members:
            pair = <PeakFragmentPair>obj
            frags.add(pair.fragment)
        return frags

    def remove_fragment(self, fragment):
        peaks = self.peaks_for(fragment)
        for peak in peaks:
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_fragment.invalidate()
        self.by_peak.invalidate()

    def remove_peak(self, peak):
        fragments = self.fragments_for(peak)
        for fragment in fragments:
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_peak.invalidate()
        self.by_fragment.invalidate()

    cpdef FragmentMatchMap copy(self):
        cdef FragmentMatchMap inst = self.__class__()
        for case in self.members:
            inst.add(case)
        return inst

    cpdef FragmentMatchMap clone(self):
        return self.copy()

    def __repr__(self):
        return "FragmentMatchMap(%s)" % (', '.join(
            f.name for f in self.fragments()),)

    cpdef clear(self):
        self.members.clear()
        self.invalidate()

    cpdef invalidate(self):
        self.by_fragment.invalidate()
        self.by_peak.invalidate()

    cpdef set peak_indices(self):
        cdef:
            PeakFragmentPair pfp
            DeconvolutedPeak peak
            set result

        result = set()
        for obj in self.members:
            pfp = <PeakFragmentPair>obj
            peak = pfp.peak
            result.add(peak._index.neutral_mass)
        return result