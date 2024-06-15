from typing import Any, List, Optional, Sequence, Set, Tuple, Union, Generic, TypeVar, DefaultDict, Iterable

from collections import defaultdict

import numpy as np

import ms_deisotope
from glycopeptidepy.structure.fragment import FragmentBase


K = TypeVar("K")
V = TypeVar("V")



class PeakFragmentPair(object):
    __slots__ = ["peak", "fragment", "fragment_name", "_hash"]

    peak: ms_deisotope.DeconvolutedPeak
    fragment: Union[FragmentBase, Any]
    fragment_name: str
    _hash: int

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

    def mass_accuracy(self):
        return (self.peak.neutral_mass - self.fragment.mass) / self.fragment.mass


class _FragmentIndexBase(Generic[K, V]):
    _mapping: DefaultDict[K, List[V]]
    fragment_set: Iterable[PeakFragmentPair]

    def __init__(self, fragment_set):
        self.fragment_set = fragment_set
        self._mapping = None

    @property
    def mapping(self):
        if self._mapping is None:
            self._create_mapping()
        return self._mapping

    def invalidate(self):
        self._mapping = None

    def __getitem__(self, key):
        return self.mapping[key]

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
        return key in self.mapping

    def __str__(self):
        return str(self.mapping)


class ByFragmentIndex(_FragmentIndexBase[Union[FragmentBase, Any], ms_deisotope.DeconvolutedPeak]):

    def _create_mapping(self):
        self._mapping = defaultdict(list)
        for peak_fragment_pair in self.fragment_set:
            self._mapping[peak_fragment_pair.fragment].append(
                peak_fragment_pair.peak)


class ByPeakIndex(_FragmentIndexBase[ms_deisotope.DeconvolutedPeak, Union[FragmentBase, Any]]):

    def _create_mapping(self):
        self._mapping = defaultdict(list)
        for peak_fragment_pair in self.fragment_set:
            self._mapping[peak_fragment_pair.peak].append(
                peak_fragment_pair.fragment)


class FragmentMatchMap(object):
    members: Set[PeakFragmentPair]
    by_fragment: ByFragmentIndex
    by_peak: ByPeakIndex

    def __init__(self):
        self.members = set()
        self.by_fragment = ByFragmentIndex(self)
        self.by_peak = ByPeakIndex(self)

    def add(self, peak: Union[ms_deisotope.DeconvolutedPeak, PeakFragmentPair],
            fragment: Optional[Union[FragmentBase, Any]]=None):
        if fragment is not None:
            peak = PeakFragmentPair(peak, fragment)
        if peak not in self.members:
            self.members.add(peak)
            self.by_peak.invalidate()
            self.by_fragment.invalidate()

    def pairs_by_name(self, name: str) -> List[PeakFragmentPair]:
        pairs = []
        for pair in self:
            if pair.fragment_name == name:
                pairs.append(pair)
        return pairs

    def fragments_for(self, peak):
        return self.by_peak[peak]

    def peaks_for(self, fragment):
        return self.by_fragment[fragment]

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

    def fragments(self):
        fragments = set()
        for peak_pair in self:
            fragments.add(peak_pair.fragment)
        return fragments

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

    def clear(self):
        self.members.clear()
        self.by_fragment.invalidate()
        self.by_peak.invalidate()


def count_peaks_shared(reference: FragmentMatchMap, alternate: FragmentMatchMap) -> Tuple[Set[int], Set[int], Set[int]]:
    ref_peaks = {p.index.neutral_mass for p in reference.by_peak.keys()}
    alt_peaks = {p.index.neutral_mass for p in alternate.by_peak.keys()}

    shared = ref_peaks & alt_peaks

    ref_peaks_ = ref_peaks - alt_peaks
    alt_peaks_ = alt_peaks - ref_peaks

    return shared, ref_peaks_, alt_peaks_


def intensity_from_peak_indices(spectrum: Sequence[ms_deisotope.DeconvolutedPeak], indices: Iterable[int]) -> float:
    return sum([spectrum[i].intensity for i in indices])


class PeakPairTransition(object):
    start: ms_deisotope.DeconvolutedPeak
    end: ms_deisotope.DeconvolutedPeak
    annotation: Any
    key: Tuple[int, int]

    _hash: int

    def __init__(self, start, end, annotation):
        self.start = start
        self.end = end
        self.annotation = annotation
        # The indices of the start peak and end peak
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
        # precompute node_sets for each path
        node_sets = {}
        for i, path in enumerate(paths):
            # track all nodes by index in this path in node_set
            node_set = set()
            for node in path:
                # add the start node index and end node index to the
                # set of nodes on this path
                node_set.update(node.key)
            node_sets[i] = (node_set, len(path))
        keep = []
        node_sets_items = list(node_sets.items())
        for i, path in enumerate(paths):
            node_set, length = node_sets[i]
            is_enclosed = False
            for key, value_pair in node_sets_items:
                # attempting to be clever here seems to cost more
                # time than it saves.
                if key == i:
                    continue
                other_node_set, other_length = value_pair
                if node_set < other_node_set:
                    is_enclosed = True
                    break
            if not is_enclosed:
                keep.append(path)
        return sorted(keep, key=len, reverse=True)

    def longest_paths(self, limit=-1):
        # get all distinct paths
        paths = []
        for ix in np.argsort(self.path_lengths())[::-1]:
            segment = self.paths_ending_at(ix)
            paths.extend(segment)
            # remove redundant paths
            paths = self.transitive_closure(paths)
            if limit > 0:
                if len(paths) > limit:
                    break
            elif len(segment) == 0:
                break
        return paths


try:
    has_c = True
    _PeakFragmentPair = PeakFragmentPair
    _PeakPairTransition = PeakPairTransition
    _SpectrumGraph = SpectrumGraph
    _FragmentMatchMap = FragmentMatchMap

    from glycresoft._c.structure.fragment_match_map import (
        PeakFragmentPair, PeakPairTransition, SpectrumGraph, FragmentMatchMap)
except ImportError:
    has_c = False
