from collections import defaultdict, namedtuple

from glycan_profiling.chromatogram_tree import Unmodified

from .spectrum_graph import ScanGraph


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

    def __init__(self):
        self.scan_map = dict()
        self.hit_map = dict()
        self.hit_to_scan_map = defaultdict(list)
        self.scan_to_hit_map = defaultdict(list)
        self.scan_hit_type_map = defaultdict(lambda: Unmodified.name)
        self._scan_graph = None

    def add_scan(self, scan):
        """Register a Scan-like object for tracking

        Parameters
        ----------
        scan : Scan-like
        """
        self.scan_map[scan.id] = scan

    def add_scan_hit(self, scan, hit, hit_type=Unmodified.name):
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

    def build_scan_graph(self, recompute=False):
        """Constructs a graph of scans where scans
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

    def compute_workloads(self):
        """Determine how much work is needed to evaluate all matches
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
        """Compute the total amount of work required to
        resolve this complete workload

        Parameters
        ----------
        workloads : list, optional
            The precomputed workload graph clusters. If None, they will
            be computed with :meth:`compute_workloads`

        Returns
        -------
        int
            The amount of work contained in this load, or 1
            if the workload is empty.
        """
        if workloads is None:
            workloads = self.compute_workloads()
        total = 1
        for m, w in workloads:
            total += w
        if total > 1:
            total -= 1
        return total

    def log_workloads(self, handle, workloads=None):
        """Writes the current workload graph cluster sizes to a text file
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
        yield self.scan_map
        yield self.hit_map
        yield self.hit_to_scan_map

    def __repr__(self):
        template = 'WorkloadManager(scan_map_size={}, hit_map_size={}, total_cross_product={})'
        rendered = template.format(
            len(self.scan_map), len(self.hit_map),
            sum(map(len, self.scan_to_hit_map.values())))
        return rendered

    _workload_batch = staticmethod(
        namedtuple("WorkloadBatch",
                   ['batch_size', 'scan_map',
                    'hit_map', 'hit_to_scan_map',
                    'scan_hit_type_map']))

    def batches(self, max_size=15e4):
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
        current_batch_size = 0
        current_scan_map = dict()
        current_hit_map = dict()
        current_hit_to_scan_map = defaultdict(list)
        current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)

        source = sorted(
            self.scan_map.items(),
            key=lambda x: x[1].precursor_information.neutral_mass)

        for scan_id, scan in source:
            current_scan_map[scan_id] = scan
            for hit_id in self.scan_to_hit_map[scan_id]:
                current_hit_map[hit_id] = self.hit_map[hit_id]
                current_hit_to_scan_map[hit_id].append(scan_id)
                current_scan_hit_type_map[
                    scan_id, hit_id] = self.scan_hit_type_map[scan_id, hit_id]
                current_batch_size += 1

            if current_batch_size > max_size:
                batch = self._workload_batch(
                    current_batch_size, current_scan_map,
                    current_hit_map, current_hit_to_scan_map,
                    current_scan_hit_type_map)
                current_batch_size = 0
                current_scan_map = dict()
                current_hit_map = dict()
                current_hit_to_scan_map = defaultdict(list)
                current_scan_hit_type_map = defaultdict(lambda: Unmodified.name)
                yield batch

        if current_batch_size > 0:
            batch = self._workload_batch(
                current_batch_size, current_scan_map,
                current_hit_map, current_hit_to_scan_map,
                current_scan_hit_type_map)
            yield batch
