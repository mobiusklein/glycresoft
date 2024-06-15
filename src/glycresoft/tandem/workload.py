import os
import warnings

from typing import (
    IO, Any, Dict, DefaultDict,
    Iterable, Iterator, List,
    Set, Tuple, Optional,
    Union, Hashable, NamedTuple
)
from collections import defaultdict

from ms_deisotope.data_source import ProcessedScan

from glycresoft.chromatogram_tree import Unmodified

from .spectrum_graph import ScanGraph


HitID = Hashable
GroupID = Hashable
DBHit = Any

WorkloadT = Union['WorkloadManager', 'WorkloadBatch']

# This number is overriden in spectrum_evaluation
# DEFAULT_WORKLOAD_MAX = 25e6
DEFAULT_WORKLOAD_MAX = -1
DEFAULT_WORKLOAD_MAX = float(os.environ.get("GLYCRESOFTDEFAULTWORKLOADMAX", DEFAULT_WORKLOAD_MAX))


class WorkloadManager(object):
    """An object that tracks the co-dependence of multiple query MSn
    scans on the same hit structures.

    By only dealing with a part of the entire workload at a time, the
    database search engine can avoid holding very large numbers of temporary
    objects in memory while waiting for related computations finish in
    random order.

    Attributes
    ----------
    hit_map : dict
        hit structure id to the structure object
    hit_to_scan_map : defaultdict(list)
        Maps hit structure id to a list of scan ids it matched
    scan_hit_type_map : defaultdict(str)
        Maps a scan id, hit structure id pair to the name of the mass
        shift type used to complete the match
    scan_map : dict
        Maps scan id to :class:`ms_deisotope.ProcessedScan` object
    scan_to_hit_map : defaultdict(list)
        Maps scan id to the structures it hit
    """

    scan_map: Dict[str, ProcessedScan]
    hit_map: Dict[HitID, DBHit]
    hit_to_scan_map: DefaultDict[HitID, List[str]]
    scan_to_hit_map: DefaultDict[str, List[HitID]]
    scan_hit_type_map: DefaultDict[Tuple[str, HitID], str]
    hit_group_map: DefaultDict[GroupID, Set[HitID]]

    _scan_graph: Optional[ScanGraph]

    def __init__(self):
        self.scan_map = dict()
        self.hit_map = dict()
        self.hit_to_scan_map = defaultdict(list)
        self.scan_to_hit_map = defaultdict(list)
        self.scan_hit_type_map = defaultdict(lambda: Unmodified.name)
        self._scan_graph = None
        self.hit_group_map = defaultdict(set)

    def clear(self):
        self.scan_map.clear()
        self.hit_map.clear()
        self.hit_to_scan_map.clear()
        self.scan_to_hit_map.clear()
        self.scan_hit_type_map.clear()
        self.hit_group_map.clear()
        self._scan_graph = None

    def pack(self):
        """
        Drop any scans that haven't been matched to a database hit.

        Returns
        -------
        WorkloadManager
        """
        unmatched_scans = set(self.scan_map) - set(self.scan_to_hit_map)
        for k in unmatched_scans:
            self.scan_map.pop(k, None)
        return self

    def update(self, other: 'WorkloadManager'):
        self.scan_map.update(other.scan_map)
        self.hit_map.update(other.hit_map)
        self.merge_hit_to_scan_map(other)
        self.merge_scan_to_hit_map(other)
        self.merge_hit_group_map(other)
        self.scan_hit_type_map.update(other.scan_hit_type_map)
        self._scan_graph = None

    def merge_hit_to_scan_map(self, other: 'WorkloadManager'):
        """Merge the :attr:`hit_to_scan_map` of another :class:`WorkloadManager`
        object into this one, adding non-redundant pairs.

        Parameters
        ----------
        other : :class:`WorkloadManager`
            The other workload manager to merge in.
        """
        for hit_id, scans in other.hit_to_scan_map.items():
            already_present = {scan.id for scan in self.hit_to_scan_map[hit_id]}
            for scan in scans:
                if scan.id not in already_present:
                    self.hit_to_scan_map[hit_id].append(scan)
                    self.scan_hit_type_map[scan.id, hit_id] = other.scan_hit_type_map[scan.id, hit_id]

    def merge_scan_to_hit_map(self, other: 'WorkloadManager'):
        """Merge the :attr:`scan_to_hit_map` of another :class:`WorkloadManager`
        object into this one, adding non-redundant pairs.

        Parameters
        ----------
        other : :class:`WorkloadManager`
            The other workload manager to merge in.
        """
        for scan_id, hit_ids in other.scan_to_hit_map.items():
            already_present = set(self.scan_to_hit_map[scan_id])
            for hit_id in hit_ids:
                if hit_id not in already_present:
                    self.scan_to_hit_map[scan_id].append(hit_id)
                    self.scan_hit_type_map[scan_id, hit_id] = other.scan_hit_type_map[scan_id, hit_id]

    def merge_hit_group_map(self, other: 'WorkloadManager'):
        for group_key, members in other.hit_group_map.items():
            self.hit_group_map[group_key].update(members)

    def add_scan(self, scan: ProcessedScan):
        """Register a Scan-like object for tracking

        Parameters
        ----------
        scan : Scan-like
        """
        self.scan_map[scan.id] = scan

    def add_scan_hit(self, scan: ProcessedScan, hit: DBHit, hit_type: str=Unmodified.name):
        """Register a matching relationship between ``scan`` and ``hit``
        with a mass shift of type ``hit_type``

        Parameters
        ----------
        scan : Scan-like
            The scan that was matched
        hit : StructureRecord-like
            The structure that was matched
        hit_type : str, optional
            The name of the :class:`.MassShift` that was used to
            bridge the total mass of the match.
        """
        self.hit_map[hit.id] = hit
        self.hit_to_scan_map[hit.id].append(scan)
        self.scan_to_hit_map[scan.id].append(hit.id)
        self.scan_hit_type_map[scan.id, hit.id] = hit_type

    def build_scan_graph(self, recompute=False) -> ScanGraph:
        """
        Constructs a graph of scans where scans
        with common hits are conneted by an edge.

        This can be used to determine the set of scans
        which all depend upon the same structures and
        so should be processed together to minimize the
        amount of work done to recompute fragments.

        Parameters
        ----------
        recompute : bool, optional
            If recompute is True, the graph will be built
            from scratch, ignoring the cached copy if it
            exists.

        Returns
        -------
        ScanGraph
        """
        if self._scan_graph is not None and not recompute:
            return self._scan_graph
        graph = ScanGraph(self.scan_map.values())
        for key, values in self.hit_to_scan_map.items():
            values = sorted(values, key=lambda x: x.index)
            for i, scan in enumerate(values):
                for other in values[i + 1:]:
                    graph.add_edge_between(scan.id, other.id, key)
        self._scan_graph = graph
        return graph

    def compute_workloads(self) -> List[Tuple]:
        """
        Determine how much work is needed to evaluate all matches
        to a set of common scans.

        Construct a dependency graph relating scans and hit
        structures using :meth:`build_scan_graph` and build connected components.

        Returns
        -------
        list
            A list of tuples, each tuple containing a connected component and
            the number of comparisons contained in that component.
        """
        graph = self.build_scan_graph()
        components = graph.connected_components()
        workloads = []
        for m in components:
            workloads.append((m, sum([len(c.edges) for c in m])))
        return workloads

    def total_work_required(self, workloads=None):
        """
        Compute the total amount of work required to
        resolve this complete workload

        Returns
        -------
        int
            The amount of work contained in this load, or 1
            if the workload is empty.
        """
        return sum(map(len, self.scan_to_hit_map.values()))

    def log_workloads(self, handle: IO[str], workloads: Optional[List['WorkloadBatch']]=None):
        """
        Writes the current workload graph cluster sizes to a text file
        for diagnostic purposes

        Parameters
        ----------
        handle : file-like
            The file to write the workload graph cluster sizes to
        workloads : list, optional
            The precomputed workload graph clusters. If None, they will
            be computed with :meth:`compute_workloads`
        """
        if workloads is None:
            workloads = self.compute_workloads()
        workloads = sorted(workloads, key=lambda x: x[0][0].precursor_information.neutral_mass,
                           reverse=True)
        for scans, load in workloads:
            handle.write("Work: %d Comparisons\n" % (load,))
            for scan_node in scans:
                handle.write("%s\t%f\t%d\n" % (
                    scan_node.id,
                    scan_node.precursor_information.neutral_mass,
                    len(scan_node.edges)))

    def __iter__(self):
        yield self.total_size
        yield self.scan_map
        yield self.hit_map
        yield self.hit_to_scan_map
        yield self.scan_hit_type_map
        yield self.hit_group_map

    def __eq__(self, other: WorkloadT):
        if self.hit_map != other.hit_map:
            return False
        elif self.scan_map != other.scan_map:
            return False
        elif self.scan_hit_type_map != other.scan_hit_type_map:
            return False
        elif self.hit_to_scan_map != other.hit_to_scan_map:
            return False
        elif self.hit_group_map != other.hit_group_map:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    @property
    def total_size(self):
        """Compute the total amount of work required to
        resolve this complete workload

        Returns
        -------
        int
            The amount of work contained in this load, or 1
            if the workload is empty.
        """
        return sum(map(len, self.scan_to_hit_map.values()))

    @property
    def scan_count(self):
        """Return the number of scans in this workload

        Returns
        -------
        int
        """
        return len(self.scan_map)

    @property
    def hit_count(self):
        """Return the number of structures in this workload

        Returns
        -------
        int
        """
        return len(self.hit_map)

    def __repr__(self):
        template = 'WorkloadManager(scan_map_size={}, hit_map_size={}, total_cross_product={})'
        rendered = template.format(
            len(self.scan_map), len(self.hit_map),
            self.total_size)
        return rendered

    def batches(self, max_size: Optional[int]=None, ignore_groups: bool=False) -> Iterator['WorkloadBatch']:
        """Partition the workload into batches of approximately
        ``max_size`` comparisons.

        This guarantee is only approximate because a batch will
        only be broken at the level of a scan.

        Parameters
        ----------
        max_size : float, optional
            The maximum size of a workload

        Yields
        ------
        WorkloadBatch
            self-contained work required for a set of related scans
            whose total work is approximately ``max_size``
        """
        if max_size is None:
            max_size = DEFAULT_WORKLOAD_MAX
        elif max_size <= 0:
            max_size = float('inf')

        if max_size == float('inf'):
            current_scan_map = self.scan_map.copy()
            current_hit_map = self.hit_map.copy()
            current_hit_to_scan_map = {k: [v.id for v in vs] for k, vs in self.hit_to_scan_map.items()}
            current_scan_hit_type_map = self.scan_hit_type_map.copy()
            current_hit_group_map = self.hit_group_map.copy()
            batch = WorkloadBatch(
                self.total_size, current_scan_map,
                current_hit_map, current_hit_to_scan_map,
                current_scan_hit_type_map, current_hit_group_map)
            if batch.batch_size:
                yield batch
            return

        current_batch_size = 0
        current_scan_map = dict()
        current_hit_map = dict()
        current_hit_to_scan_map = defaultdict(list)
        current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)
        current_hit_group_map = defaultdict(set)

        if self.hit_group_map and not ignore_groups:
            source = sorted(self.hit_group_map.items())
            batch_index = 0
            for group_id, hit_ids in source:
                current_hit_group_map[group_id] = hit_ids
                for hit_id in hit_ids:
                    current_hit_map[hit_id] = self.hit_map[hit_id]
                    for scan in self.hit_to_scan_map[hit_id]:
                        scan_id = scan.id
                        current_scan_map[scan_id] = self.scan_map[scan_id]
                        current_hit_to_scan_map[hit_id].append(scan_id)
                        current_scan_hit_type_map[
                            scan_id, hit_id] = self.scan_hit_type_map[scan_id, hit_id]
                        current_batch_size += 1

                if current_batch_size > max_size:
                    batch = WorkloadBatch(
                        current_batch_size, current_scan_map,
                        current_hit_map, current_hit_to_scan_map,
                        current_scan_hit_type_map, current_hit_group_map)
                    if batch.batch_size / max_size > 2:
                        warnings.warn("Batch %d has size %d, %0.3f%% larger than threshold" % (
                            batch_index, batch.batch_size, batch.batch_size * 100. / max_size))
                    batch_index += 1
                    current_batch_size = 0
                    current_scan_map = dict()
                    current_hit_map = dict()
                    current_hit_to_scan_map = defaultdict(list)
                    current_scan_hit_type_map = defaultdict(
                        lambda: Unmodified.name)
                    current_hit_group_map = defaultdict(set)
                    if batch.batch_size:
                        yield batch
        else:
            source = sorted(
                self.scan_map.items(),
                key=lambda x: x[1].precursor_information.neutral_mass)
            batch_index = 0
            for scan_id, scan in source:
                current_scan_map[scan_id] = scan
                for hit_id in self.scan_to_hit_map[scan_id]:
                    current_hit_map[hit_id] = self.hit_map[hit_id]
                    current_hit_to_scan_map[hit_id].append(scan_id)
                    current_scan_hit_type_map[
                        scan_id, hit_id] = self.scan_hit_type_map[scan_id, hit_id]
                    current_batch_size += 1

                if current_batch_size > max_size:
                    batch = WorkloadBatch(
                        current_batch_size, current_scan_map,
                        current_hit_map, current_hit_to_scan_map,
                        current_scan_hit_type_map, current_hit_group_map)
                    if batch.batch_size / max_size > 2:
                        warnings.warn("Batch %d has size %d, %0.3f%% larger than threshold" % (
                            batch_index, batch.batch_size, batch.batch_size * 100. / max_size))
                    batch_index += 1
                    current_batch_size = 0
                    current_scan_map = dict()
                    current_hit_map = dict()
                    current_hit_to_scan_map = defaultdict(list)
                    current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)
                    current_hit_group_map = defaultdict(set)
                    if batch.batch_size:
                        yield batch

        if current_batch_size > 0:
            batch = WorkloadBatch(
                current_batch_size, current_scan_map,
                current_hit_map, current_hit_to_scan_map,
                current_scan_hit_type_map, current_hit_group_map)
            if batch.batch_size:
                yield batch

    @classmethod
    def merge(cls, workloads: Iterable[WorkloadT]):
        total = cls()
        for inst in workloads:
            total.update(inst)
        return total

    def mass_range(self) -> Tuple[float, float]:
        lo = float('inf')
        hi = 0
        for scan in self.scan_map.values():
            mass = scan.precursor_information.neutral_mass
            if mass < lo:
                lo = mass
            if mass > hi:
                hi = mass
        return (lo, hi)


class WorkloadBatch(NamedTuple):

    batch_size: int
    scan_map: Dict[str, ProcessedScan]
    hit_map: Dict[HitID, DBHit]
    hit_to_scan_map: DefaultDict[HitID, List[str]]
    scan_hit_type_map: DefaultDict[Tuple[str, HitID], str]
    hit_group_map: DefaultDict[GroupID, Set[HitID]]

    def clear(self):
        self.scan_map.clear()
        self.hit_map.clear()
        self.hit_to_scan_map.clear()
        self.scan_hit_type_map.clear()
        self.hit_group_map.clear()
