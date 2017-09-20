from collections import defaultdict

import numpy as np


class PeakFragmentPair(object):
    __slots__ = ["peak", "fragment", "fragment_name", "_hash"]

    def __init__(self, peak, fragment):
        self.peak = peak
        self.fragment = fragment
        self.fragment_name = fragment.name
        self._hash = hash(self.peak)

    def __eq__(self, other):
        return (self.peak == other.peak) and (self.fragment_name == other.fragment_name)

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


class FragmentMatchMap(object):
    def __init__(self):
        self.members = set()
        self.by_fragment = defaultdict(list)
        self.by_peak = defaultdict(list)

    def add(self, peak, fragment=None):
        if fragment is not None:
            peak = PeakFragmentPair(peak, fragment)
        if peak not in self.members:
            self.members.add(peak)
            self.by_fragment[peak.fragment].append(peak.peak)
            self.by_peak[peak.peak].append(peak.fragment)

    def fragments_for(self, peak):
        return self.by_peak[peak]

    def peaks_for(self, fragment):
        return self.by_fragment[fragment]

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

    def fragments(self):
        frags = set()
        for peak, fragment in self:
            frags.add(fragment)
        return iter(frags)

    def remove_fragment(self, fragment):
        peaks = self.peaks_for(fragment)
        for peak in peaks:
            fragments_from_peak = self.by_peak[peak]
            kept = [f for f in fragments_from_peak if f != fragment]
            if len(kept) == 0:
                self.by_peak.pop(peak)
            else:
                self.by_peak[peak] = kept
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_fragment.pop(fragment)

    def remove_peak(self, peak):
        fragments = self.fragments_for(peak)
        for fragment in fragments:
            peaks_from_fragment = self.by_fragment[fragment]
            kept = [p for p in peaks_from_fragment if p != peak]
            if len(kept) == 0:
                self.by_fragment.pop(fragment)
            else:
                self.by_fragment[fragment] = kept
            self.members.remove(PeakFragmentPair(peak, fragment))
        self.by_peak.pop(peak)

    def copy(self):
        inst = self.__class__()
        for case in self.members:
            inst.add(case)
        return inst

    def clone(self):
        return self.copy()

    def __repr__(self):
        return "FragmentMatchMap(%s)" % (', '.join(
            f.name for f in self.fragments()),)


class PeakPairTransition(object):
    def __init__(self, start, end, annotation):
        self.start = start
        self.end = end
        self.annotation = annotation
        self.key = (self.start.index.neutral_mass, self.end.index.neutral_mass)
        self._hash = hash(self.key)

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


class SpectrumGraph(object):
    def __init__(self):
        self.transitions = set()
        self.by_first = defaultdict(list)
        self.by_second = defaultdict(list)

    def add(self, p1, p2, annotation):
        p1, p2 = sorted((p1, p2), key=lambda x: x.index.neutral_mass)
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

    def topological_sort(self, adjacency_matrix=None):
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

    def paths_starting_at(self, ix):
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

    def paths_ending_at(self, ix):
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
                    extended_paths.append([trans] + path)
            paths = extended_paths
            if len(paths) == 0:
                break
        return self.transitive_closure(finished_paths)

    def transitive_closure(self, paths):
        node_sets = {}
        for i, path in enumerate(paths):
            node_set = set()
            for node in path:
                node_set.update(node.key)
            node_sets[i] = (node_set, len(path))
        keep = []
        for i, path in enumerate(paths):
            node_set, length = node_sets[i]
            is_enclosed = False
            for key, value_pair in node_sets.items():
                if key == i:
                    continue
                other_node_set, other_length = value_pair
                if node_set < other_node_set:
                    is_enclosed = True
                    break
            if not is_enclosed:
                keep.append(path)
        return sorted(keep, key=len, reverse=True)

    def longest_paths(self):
        # get all distinct paths
        paths = []
        for ix in np.nonzero(self.path_lengths())[0]:
            paths.extend(self.paths_ending_at(ix))
        # remove redundant paths
        paths = self.transitive_closure(paths)
        return paths
