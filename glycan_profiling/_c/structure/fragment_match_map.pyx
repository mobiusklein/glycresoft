# cython: embedsignature=True
from collections import defaultdict

cimport cython
from cpython cimport PyList_Append, PyList_Size, PyList_GetItem

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak

np.import_array()


@cython.freelist(1000000)
cdef class PeakFragmentPair(object):

    cdef:
        public DeconvolutedPeak peak
        public object fragment
        public str fragment_name
        public long _hash

    def __init__(self, peak, fragment):
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.name
        self._hash = hash(self.peak)

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


@cython.freelist(1000000)
cdef class PeakPairTransition(object):
    cdef:
        public DeconvolutedPeak start
        public DeconvolutedPeak end
        public object annotation
        public tuple key
        public long _hash

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
        if self.key != other.key:
            return False
        elif self.start != other.start:
            return False
        elif self.end != other.end:
            return False
        elif self.annotation != other.annotation:
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

    cdef:
        public set transitions
        public object by_first
        public object by_second

    def __init__(self):
        self.transitions = set()
        self.by_first = defaultdict(list)
        self.by_second = defaultdict(list)

    cpdef add(self, DeconvolutedPeak p1, DeconvolutedPeak p2, object annotation):
        if p1._index.neutral_mass > p2._index.neutral_mass:
            temp = p2
            p2 = p1
            p1 = temp
        trans = PeakPairTransition(p1, p2, annotation)
        self.transitions.add(trans)
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

    cpdef list paths_starting_at(self, ix):
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
        return self.transitive_closure(finished_paths)

    cpdef list paths_ending_at(self, ix):
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
        return self.transitive_closure(finished_paths)

    cpdef list transitive_closure(self, list paths):
        cdef:
            list path
            list node_sets, keep, node_sets_items
            set node_set, other_node_set
            PeakPairTransition node
            size_t i, j
            bint is_enclosed

        # precompute node_sets for each path
        node_sets = []
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
                    node_set.add(node.start._index.neutral_mass)
                node_set.add(node.end._index.neutral_mass)
            node_sets.append(node_set)

        keep = []
        for i in range(PyList_Size(paths)):
            path = <list>PyList_GetItem(paths, i)
            is_enclosed = False
            node_set = <set>PyList_GetItem(node_sets, i)
            for j in range(PyList_Size(paths)):
                if i == j:
                    continue
                other_node_set = <set>PyList_GetItem(node_sets, j)
                if node_set < other_node_set:
                    is_enclosed = True
                    break
            if not is_enclosed:
                keep.append(path)
        return sorted(keep, key=len, reverse=True)

    def longest_paths(self, int limit=-1):
        # get all distinct paths
        paths = []
        for ix in np.nonzero(self.path_lengths())[0]:
            paths.extend(self.paths_ending_at(ix))
            # remove redundant paths
            paths = self.transitive_closure(paths)
            if limit > 0:
                if len(paths) > limit:
                    break
        # paths = self.transitive_closure(paths)
        return paths
