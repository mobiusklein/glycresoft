import math

import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterator, List, Set, Tuple, TYPE_CHECKING, Optional

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase

from ms_deisotope import DeconvolutedPeak
from ms_deisotope.data_source import ProcessedScan

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, HashableGlycanComposition

from glycresoft.chromatogram_tree import Unmodified

from .fragment_match_map import SpectrumGraph, PeakPairTransition

if TYPE_CHECKING:
    from glycresoft.database.mass_collection import NeutralMassDatabase

hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
xylose = FrozenMonosaccharideResidue.from_iupac_lite("Xyl")
fucose = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")
neuac = FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")
neugc = FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")


def collect_glycan_composition_from_annotations(path: 'Path') -> HashableGlycanComposition:
    gc = HashableGlycanComposition()
    for edge in path:
        if isinstance(edge.annotation, tuple):
            for part in edge.annotation:
                gc[part] += 1
        else:
            gc[edge.annotation] += 1
    if path.end_node is not None:
        if isinstance(path.end_node, tuple):
            for part in path.end_node:
                gc[part] += 1
        else:
            gc[path.end_node] += 1
    return gc


class MassWrapper(object):
    '''An adapter class to make types whose mass calculation is a method
    (:mod:`glypy` dynamic graph components) compatible with code where the
    mass calculation is an  attribute (:mod:`glycopeptidepy` objects and
    most things here)

    Hashes and compares as :attr:`obj`

    Attributes
    ----------
    obj: object
        The wrapped object
    mass: float
        The mass of :attr:`obj`
    '''
    obj: Any
    mass: float

    def __init__(self, obj, mass=None):
        self.obj = obj
        if mass is not None:
            self.mass = mass
        else:
            try:
                # object's mass is a method
                self.mass = obj.mass()
            except TypeError:
                # object's mass is a plain attribute
                self.mass = obj.mass

    def __repr__(self):
        if isinstance(self.obj, tuple):
            filling = ', '.join(tuple(map(str, self.obj)))
        else:
            filling = str(self.obj)

        return f"{self.__class__.__name__}({filling})"

    def __eq__(self, other):
        return self.obj == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.obj)

    def __lt__(self, other):
        return self.mass < other.mass

    def __gt__(self, other):
        return self.mass > other.mass


default_components = (hexnac, hexose, xylose, fucose, neuac, neugc)


class PeakGroup(object):
    """
    An adapter for a collection of :class:`~.DeconvolutedPeak` objects which share
    the same approximate neutral mass that looks like a single :class:`~.DeconvolutedPeak`
    object for edge-like calculations with :class:`EdgeGroup`.

    Attributes
    ----------
    peaks: tuple
        A tuple of the *unique* peaks forming this group
    neutral_mass: float
        The intensity weighted average neutral mass of the peaks in this group
    intensity: float
        The sum of the intensities of the peaks in this group
    """
    peaks: Tuple[DeconvolutedPeak]
    neutral_mass: float
    intensity: float

    def __init__(self, peaks):
        self.peaks = tuple(sorted(set(peaks), key=lambda x: (x.neutral_mass, x.charge)))
        self.neutral_mass = 0.0
        self.intensity = 0.0
        self._hash = hash((p.index.neutral_mass for p in self.peaks))
        self._initialize()

    def _initialize(self):
        neutral_mass_acc = 0.0
        intensity = 0.0
        for peak in self.peaks:
            neutral_mass_acc += peak.neutral_mass * peak.intensity
            intensity += peak.intensity
        self.intensity = intensity
        self.neutral_mass = neutral_mass_acc / intensity

    def __eq__(self, other):
        return self.peaks == other.peaks

    def __hash__(self):
        return self._hash

    def __getitem__(self, i: int) -> DeconvolutedPeak:
        return self.peaks[i]

    def __len__(self):
        return len(self.peaks)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.neutral_mass}, {self.intensity}, {size})"
        return template.format(self=self, size=len(self))


class EdgeGroup(object):
    """
    An adapter for a collection of :class:`~.PeakPairTransition` objects which all belong to
    different :class:`Path` objects that share the same annotation sequence and approximate start
    and end mass.

    Attributes
    ----------
    start: PeakGroup
        The peaks that make up the starting mass for this edge group
    end: PeakGroup
        The peaks that make up the ending mass for this edge group
    annotation: object
        The common annotation for members of this group
    """

    start: PeakGroup
    annotation: Any
    end: PeakGroup

    _transitions: List[PeakPairTransition]
    _hash: int

    def __init__(self, transitions):
        self._transitions = transitions
        self.start = None
        self.end = None
        self._hash = None
        self.annotation = self._transitions[0].annotation
        self._initialize_by_transitions()

    def __hash__(self):
        return self._hash

    def __eq__(self, other: 'EdgeGroup'):
        if other is None:
            return False
        return self.start == other.start and self.end == other.end and self.annotation == other.annotation

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, i):
        return self._transitions[i]

    def __len__(self):
        return len(self._transitions)

    def _initialize_by_transitions(self):
        if self._transitions:
            starts = [edge.start for edge in self._transitions]
            ends = [edge.end for edge in self._transitions]
            self.start = PeakGroup(starts)
            self.end = PeakGroup(ends)
            self._hash = hash((self.start.peaks[0].index.neutral_mass,
                               self.end.peaks[0].index.neutral_mass))
        else:
            self._hash = hash(())

    @classmethod
    def from_combination(cls, start: PeakGroup, end: PeakGroup, annotation: MassWrapper):
        return cls([PeakPairTransition(start_peak, end_peak, annotation)
                    for start_peak, end_peak in itertools.product(start, end)])

    @classmethod
    def aggregate(cls, paths):
        edges = []
        n = len(paths)
        if n == 0:
            return []
        m = len(paths[0])
        for i in range(m):
            group = []
            for j in range(n):
                group.append(paths[j][i])
            edges.append(group)
        return paths[0].__class__([cls(g) for g in edges])

    def __repr__(self):
        return ("{self.__class__.__name__}({self.start.neutral_mass:0.3f} "
                "-{self.annotation}-> {self.end.neutral_mass:0.3f})").format(self=self)


def collect_paths(paths: List['Path'], error_tolerance: float=1e-5) -> List[List['Path']]:
    """
    Group together paths which share the same annotation sequence and approximate
    start and end masses.

    Parameters
    ----------
    paths: :class:`list`
        A list of :class:`Path` objects to group

    Returns
    -------
    :class:`list` of :class:`list` of :class:`Path` objects
    """
    groups = defaultdict(list)
    if not paths:
        return []
    for path in paths:
        key = tuple(e.annotation for e in path)
        groups[key].append(path)
    result = []
    for key, block_members in groups.items():
        block_members = sorted(block_members, key=lambda x: x.start_mass)
        current_path = block_members[0]
        members = [current_path]
        for path in block_members[1:]:
            if abs(current_path.start_mass - path.start_mass) / path.start_mass < error_tolerance:
                members.append(path)
            else:
                result.append(members)
                current_path = path
                members = [current_path]
        result.append(members)
    return result


def score_path(path: 'Path') -> float:
    if not path:
        return 0
    acc = math.log10(path.transitions[0].start.intensity)
    for edge in path.transitions:
        acc += math.log10(edge.end.intensity)
    return acc


class Path(object):
    """
    Represent a single contiguous sequence of :class:`~.PeakPairTransition` objects
    forming a sequence tag, or a path along the edges of a peak graph.

    Attributes
    ----------
    transitions: :class:`list` of :class:`~.PeakPairTransition`
        The edges connecting pairs of peaks.
    total_signal: float
        The total signal captured by this path
    start_mass: float
        The lowest mass among all peaks in the path
    end_mass: float
        The highest mass among all peaks in the path
    """

    transitions: List[PeakPairTransition]
    total_signal: float
    start_mass: float
    end_mass: float

    start_node: Optional[Any]
    end_node: Optional[Any]

    _peaks_used: Optional[Set[DeconvolutedPeak]]
    _edges_used: Optional[DefaultDict[Tuple[DeconvolutedPeak, DeconvolutedPeak], List[Any]]]

    def __init__(self, edge_list, start_node=None, end_node=None):
        self.transitions = edge_list
        self.total_signal = self._total_signal()
        self.start_mass = self[0].start.neutral_mass
        self.end_mass = self[-1].end.neutral_mass
        self.start_node = start_node
        self.end_node = end_node
        self._peaks_used = None
        self._edges_used = None

    def copy(self) -> 'Path':
        return self.__class__(self.transitions, self.start_node, self.end_node)

    @property
    def peaks(self):
        if self._peaks_used is None:
            self._peaks_used = self._build_peaks_set()
        return self._peaks_used

    def _build_peaks_set(self) -> PeakPairTransition:
        peaks = set()
        for edge in self:
            peaks.add(edge.end)
        peaks.add(self[0].start)
        return peaks

    def _build_edges_used(self) -> DefaultDict[Tuple[DeconvolutedPeak, DeconvolutedPeak], List[Any]]:
        mapping = defaultdict(list)
        for edge in self:
            mapping[edge.start, edge.end].append(edge)
        return mapping

    def __contains__(self, edge: PeakPairTransition):
        return self.has_edge(edge, True)

    def has_edge(self, edge: PeakPairTransition, match_annotation: bool=False) -> bool:
        if self._edges_used is None:
            self._edges_used = self._build_edges_used()
        key = (edge.start, edge.end)
        if key in self._edges_used:
            edges = self._edges_used[key]
            if match_annotation:
                for e in edges:
                    if e == edge:
                        return True
                return False
            else:
                for e in edges:
                    if e.key != edge.key:
                        continue
                    if e.start != edge.start:
                        continue
                    if e.end != edge.end:
                        continue
                    return True
                return False

    def has_peak(self, peak: DeconvolutedPeak) -> bool:
        if self._peaks_used is None:
            self._peaks_used = self._build_peaks_set()
        return peak in self._peaks_used

    def has_peaks(self, peaks_set: Set[DeconvolutedPeak]) -> bool:
        if self._peaks_used is None:
            self._peaks_used = self._build_peaks_set()
        return peaks_set & self._peaks_used

    def __iter__(self) -> Iterator[PeakPairTransition]:
        return iter(self.transitions)

    def __getitem__(self, i):
        return self.transitions[i]

    def __len__(self):
        return len(self.transitions)

    def _total_signal(self) -> float:
        total = 0.0
        for edge in self:
            total += edge.end.intensity
        total += self[0].start.intensity
        return total

    def __repr__(self):
        transitions = '->'.join(_format_annotation(e.annotation)
                                for e in self)
        if self.start_node is not None:
            transitions = f"{_format_annotation(self.start_node)}:-{transitions}"
        if self.end_node is not None:
            transitions = f"{transitions}-:{_format_annotation(self.end_node)}"
        return f"{self.__class__.__name__}({transitions}, {self.total_signal:0.4e}, {self.start_mass}, {self.end_mass})"


def _format_annotation(annot: Any):
    return '[%s]' % ', '.join(map(str, annot)) if isinstance(annot, tuple) else str(annot)


class PathSet(Sequence):
    def __init__(self, paths, ordered=False):
        self.paths = (sorted(paths, key=lambda x: x.start_mass)
                      if not ordered else paths)

    def __getitem__(self, i):
        return self.paths[i]

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.paths)[1:-1])

    def _repr_pretty_(self, p, cycle):
        with p.group(9, "%s([" % self.__class__.__name__, "])"):
            for i, path in enumerate(self):
                if i:
                    p.text(",")
                    p.breakable()
                p.pretty(path)

    def is_path_disjoint(self, path):
        for p in self:
            if p.has_peaks(path.peaks):
                return False
        return True

    def threshold(self, topn=100):
        return self.__class__(sorted(self, key=lambda x: x.total_signal, reverse=True)[:topn])


class PathFinder(object):
    components: List[MassWrapper]
    product_error_tolerance: float
    merge_paths: bool
    precursor_error_tolerance: float
    gap_completion_lookup: Dict[int, 'NeutralMassDatabase[MassWrapper]']
    _gap_completion_mass_range: Dict[int, Tuple[float, float]]

    def __init__(self, components=None,
                 product_error_tolerance: float=1e-5,
                 merge_paths: bool=True,
                 precursor_error_tolerance: float=1e-5):
        if components is None:
            components = default_components
        self.components = list(map(MassWrapper, components))
        self.product_error_tolerance = product_error_tolerance
        self.merge_paths = merge_paths
        self.precursor_error_tolerance = precursor_error_tolerance
        self.gap_completion_lookup = {}
        self._gap_completion_mass_range = {}

    def build_graph(self, scan: ProcessedScan, mass_shift: MassShiftBase=Unmodified) -> SpectrumGraph:
        graph = SpectrumGraph()
        has_tandem_shift = abs(mass_shift.tandem_mass) > 0
        for peak in scan.deconvoluted_peak_set:
            for component in self.components:
                for other_peak in scan.deconvoluted_peak_set.all_peaks_for(
                        peak.neutral_mass + component.mass, self.product_error_tolerance):
                    graph.add(peak, other_peak, component.obj)
                if has_tandem_shift:
                    for other_peak in scan.deconvoluted_peak_set.all_peaks_for(
                            peak.neutral_mass + component.mass + mass_shift.tandem_mass,
                            self.product_error_tolerance):
                        graph.add(peak, other_peak, component.obj)
        return graph

    def paths_from_graph(self, graph: SpectrumGraph, limit: int=200) -> List[Path]:
        paths = []
        min_start_mass = min(c.mass for c in self.components) + 1
        for path in graph.longest_paths(limit=limit):
            path = Path(path)
            if path.start_mass < min_start_mass:
                continue
            paths.append(path)
        return paths

    def aggregate_paths(self, paths: List['Path']) -> DefaultDict[Tuple[Any], List['Path']]:
        groups = defaultdict(list)
        for path in paths:
            key = tuple(e.annotation for e in path)
            groups[key].append(path)
        return groups

    def collect_paths(self, paths: List['Path'], error_tolerance: float = 1e-5) -> List[List['Path']]:
        """
        Group together paths which share the same annotation sequence and approximate
        start and end masses.

        Parameters
        ----------
        paths: :class:`list`
            A list of :class:`Path` objects to group

        Returns
        -------
        :class:`list` of :class:`list` of :class:`Path` objects
        """
        if not paths:
            return []
        groups = self.aggregate_paths(paths)
        result = []
        for _key, block_members in groups.items():
            block_members = sorted(block_members, key=lambda x: x.start_mass)
            current_path = block_members[0]
            members = [current_path]
            for path in block_members[1:]:
                if abs(current_path.start_mass - path.start_mass) / path.start_mass < error_tolerance:
                    members.append(path)
                else:
                    result.append(members)
                    current_path = path
                    members = [current_path]
            result.append(members)
        result = [EdgeGroup.aggregate(block) for block in result]
        return result

    def paths(self, scan: ProcessedScan, mass_shift: MassShiftBase=Unmodified, limit: int=200) -> List[Path]:
        graph = self.build_graph(scan, mass_shift)
        paths = self.paths_from_graph(graph, limit)
        if self.merge_paths:
            paths = self.collect_paths(paths, self.precursor_error_tolerance)
        return paths

    def _fill_gap_size_bin(self, size: int):
        combos = []
        for items in  itertools.combinations_with_replacement(self.components, size):
            step = MassWrapper(tuple(v.obj for v in items), sum([v.mass for v in items]))
            combos.append(step)
        from glycresoft.database.mass_collection import NeutralMassDatabase
        return NeutralMassDatabase(combos, lambda x: x.mass)

    def gap_bin_entries(self, size: int) -> 'NeutralMassDatabase[MassWrapper]':
        if size not in self.gap_completion_lookup:
            self.gap_completion_lookup[size] = self._fill_gap_size_bin(size)
            masses = [m.mass for m in self.gap_completion_lookup[size]]
            self._gap_completion_mass_range[size] = (min(masses), max(masses))
        return self.gap_completion_lookup[size]

    def complete_paths(self, paths: List[Path], max_gap_size: int = 2) -> List[Path]:
        from glycresoft.database.mass_collection import NeutralMassDatabase
        by_front = NeutralMassDatabase(paths, lambda x: x.start_mass)
        by_end = NeutralMassDatabase(paths, lambda x: x.end_mass)
        max_gap_size = 2
        extensions = []

        for path in by_end:
            for i in range(1, max_gap_size + 1):
                spacers = self.gap_bin_entries(i)
                for candidate in spacers:
                    suffixes = by_front.search_mass_ppm(
                        path.end_mass + candidate.mass, self.product_error_tolerance)
                    for suffix in suffixes:
                        extensions.append(Path(
                                list(path) + [EdgeGroup.from_combination(
                                path[-1].end, suffix[0].start, candidate.obj)] + list(suffix),
                                path.start_node,
                                suffix.end_node,
                            ))
        return extensions + paths

    def complete_precursor(self, scan: ProcessedScan,
                           path: Path,
                           mass_shift: MassShiftBase=Unmodified,
                           max_gap_size: int=3) -> List[Path]:
        precursor_mass = scan.precursor_information.neutral_mass
        leftover_mass: float = precursor_mass - (path.end_mass + mass_shift.mass)
        candidates: List[MassWrapper] = []
        for i in range(1, max_gap_size + 1):
            candidates.extend(
                self.gap_bin_entries(i).search_mass(
                leftover_mass, error_tolerance=0.2))
        result = []
        for candidate in candidates:
            proposed_mass = candidate.mass + path.end_mass + mass_shift.mass
            if abs(proposed_mass - precursor_mass) / precursor_mass <= self.precursor_error_tolerance:
                completed = path.copy()
                completed.end_node = candidate.obj
                result.append(completed)
        if not result:
            result = [path]
        return result


