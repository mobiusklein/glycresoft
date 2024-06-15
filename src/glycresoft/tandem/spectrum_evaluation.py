import os
import functools

from typing import List, Mapping, Optional, TYPE_CHECKING

from glycresoft.chromatogram_tree.mass_shift import MassShift

from ms_deisotope import isotopic_shift
from ms_deisotope.data_source import ProcessedScan

from glycresoft.task import TaskBase
from glycresoft.chromatogram_tree import Unmodified


from .process_dispatcher import (
    IdentificationProcessDispatcher,
    SolutionHandler,
    MultiScoreSolutionHandler,
    MultiScoreSolutionPacker,
    SequentialIdentificationProcessor)

from .workload import WorkloadManager, DEFAULT_WORKLOAD_MAX
from .spectrum_match import (
    MultiScoreSpectrumSolutionSet,
    SpectrumSolutionSet)


if TYPE_CHECKING:
    from glycresoft.database.mass_collection import SearchableMassCollection


def group_by_precursor_mass(scans, window_size=1.5e-5):
    scans = sorted(
        scans, key=lambda x: x.precursor_information.extracted_neutral_mass,
        reverse=True)
    groups = []
    if len(scans) == 0:
        return groups
    current_group = [scans[0]]
    last_scan = scans[0]
    for scan in scans[1:]:
        delta = abs(
            (scan.precursor_information.extracted_neutral_mass -
             last_scan.precursor_information.extracted_neutral_mass
             ) / last_scan.precursor_information.extracted_neutral_mass)
        if delta > window_size:
            groups.append(current_group)
            current_group = [scan]
        else:
            current_group.append(scan)
        last_scan = scan
    groups.append(current_group)
    return groups


class TandemClusterEvaluatorBase(TaskBase):
    tandem_cluster: List[ProcessedScan]
    structure_database: "SearchableMassCollection"
    verbose: bool
    n_processes: int
    mass_shifts: List[MassShift]
    probing_range_for_missing_precursors: int
    mass_shift_map: Mapping[str, MassShift]
    batch_size: int
    trust_precursor_fits: bool

    neutron_offset = isotopic_shift()
    solution_set_type = SpectrumSolutionSet

    def __init__(self, tandem_cluster, scorer_type, structure_database, verbose=False,
                 n_processes=1, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None, batch_size=DEFAULT_WORKLOAD_MAX, trust_precursor_fits=True):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.verbose = verbose
        self.n_processes = n_processes
        self.ipc_manager = ipc_manager
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.mass_shifts = mass_shifts
        self.mass_shift_map = {
            m.name: m for m in self.mass_shifts
        }
        self.batch_size = batch_size
        self.trust_precursor_fits = trust_precursor_fits

    def _mark_hit(self, match):
        return self.structure_database.mark_hit(match)

    def _mark_batch(self):
        return self.structure_database.mark_batch()

    def search_database_for_precursors(self, mass: float, precursor_error_tolerance: float=1e-5):
        return self.structure_database.search_mass_ppm(mass, precursor_error_tolerance)

    def find_precursor_candidates(self, scan: ProcessedScan, error_tolerance: float=1e-5,
                                  probing_range: int=0, mass_shift: Optional[MassShift]=None):
        if mass_shift is None:
            mass_shift = Unmodified
        hits = []
        intact_mass = scan.precursor_information.extracted_neutral_mass
        for i in range(probing_range + 1):
            query_mass = intact_mass - (i * self.neutron_offset) - mass_shift.mass
            hits.extend(
                map(self._mark_hit,
                    self.search_database_for_precursors(query_mass, error_tolerance)))
        return hits

    def score_one(self, scan: ProcessedScan, precursor_error_tolerance=1e-5,
                  mass_shifts: List[MassShift]=None, **kwargs) -> SpectrumSolutionSet:
        """Search one MSn scan against the database and score all candidate matches

        Parameters
        ----------
        scan : ms_deisotope.ProcessedScan
            The MSn scan to search
        precursor_error_tolerance : float, optional
            The mass error tolerance for the precursor
        mass_shifts : Iterable of MassShift, optional
            The set of MassShifts to consider. Defaults to (Unmodified,)
        **kwargs
            Keyword arguments passed to :meth:`evaluate`

        Returns
        -------
        SpectrumSolutionSet
            The set of solutions for the searched scan
        """
        if mass_shifts is None:
            mass_shifts = (Unmodified,)
        solutions = []

        if (not scan.precursor_information.defaulted and self.trust_precursor_fits):
            probe = 0
        else:
            probe = self.probing_range_for_missing_precursors
        hits = []
        for mass_shift in mass_shifts:
            hits.extend(self.find_precursor_candidates(
                scan, precursor_error_tolerance, probing_range=probe,
                mass_shift=mass_shift))
            self._mark_batch()
            for structure in hits:
                result = self.evaluate(
                    scan, structure, mass_shift=mass_shift, **kwargs)
                solutions.append(result)
        out = self.solution_set_type(
            scan, solutions).sort().select_top()
        return out

    def score_all(self, precursor_error_tolerance: float=1e-5,
                  simplify: bool=False, **kwargs) -> List[SpectrumSolutionSet]:
        """Search and score all scans in :attr:`tandem_cluster`

        Parameters
        ----------
        precursor_error_tolerance : float, optional
            The mass error tolerance for the precursor
        simplify : bool, optional
            Whether or not to call :meth:`.SpectrumSolutionSet.simplify` on each
            scan's search result.
        **kwargs
            Keyword arguments passed to :meth:`evaluate`

        Returns
        -------
        list of SpectrumSolutionSet
        """
        out = []

        # loop over mass_shifts so that sequential queries deals with similar masses to
        # make better use of the database's cache
        for mass_shift in sorted(self.mass_shifts, key=lambda x: x.mass):
            # make iterable for score_one API
            mass_shift = (mass_shift,)
            for scan in self.tandem_cluster:
                solutions = self.score_one(
                    scan, precursor_error_tolerance,
                    mass_shifts=mass_shift,
                    **kwargs)
                if len(solutions) > 0:
                    out.append(solutions)
        if simplify:
            for case in out:
                case.simplify()
                case.select_top()
        return out

    def evaluate(self, scan: ProcessedScan, structure, *args, **kwargs):
        """Computes the quality of the match between ``scan``
        and the component fragments of ``structure``.

        To be overridden by implementing classes.

        Parameters
        ----------
        scan : ms_deisotope.ProcessedScan
            The scan to evaluate
        structure : object
            An object representing a structure (peptide, glycan, glycopeptide)
            to produce fragments from and search against ``scan``
        *args
            Propagated
        **kwargs
            Propagated

        Returns
        -------
        SpectrumMatcherBase
        """
        raise NotImplementedError()

    def _map_scans_to_hits(self, scans: List[ProcessedScan], precursor_error_tolerance: float=1e-5) -> WorkloadManager:
        groups = group_by_precursor_mass(
            scans, precursor_error_tolerance * 1.5)

        workload = WorkloadManager()

        i = 0
        n = len(scans)
        report_interval = 0.25 * n if n < 2000 else 0.1 * n
        last_report = report_interval
        self.log("... Begin Collecting Hits")
        for mass_shift in sorted(self.mass_shifts, key=lambda x: x.mass):
            self.log("... Mapping For %s" % (mass_shift.name,))
            i = 0
            last_report = report_interval
            for group in groups:
                if len(group) == 0:
                    continue
                i += len(group)
                report = False
                if i > last_report:
                    report = True
                    self.log(
                        "...... Mapping %0.2f%% of spectra (%d/%d) %0.4f" % (
                            i * 100. / n, i, n,
                            group[0].precursor_information.extracted_neutral_mass))
                    while last_report < i and report_interval != 0:
                        last_report += report_interval
                j = 0
                for scan in group:
                    workload.add_scan(scan)

                    # For a sufficiently dense database or large value of probe, this
                    # could easily throw the mass interval cache scheme into hysteresis.
                    # If this does occur, instead of doing this here, we could search all
                    # defaulted precursors afterwards.
                    if not scan.precursor_information.defaulted and self.trust_precursor_fits:
                        probe = 0
                    else:
                        probe = self.probing_range_for_missing_precursors
                    hits = self.find_precursor_candidates(
                        scan, precursor_error_tolerance, probing_range=probe,
                        mass_shift=mass_shift)
                    for hit in hits:
                        j += 1
                        workload.add_scan_hit(scan, hit, mass_shift.name)
                if report:
                    self.log("...... Mapping Segment Done. (%d spectrum-pairs)" % (j,))
        self._mark_batch()
        return workload

    def _get_solution_handler(self):
        if issubclass(self.solution_set_type, MultiScoreSpectrumSolutionSet):
            handler_tp = MultiScoreSolutionHandler
            score_set_type = self.scorer_type.get_score_set_type()
            packer = MultiScoreSolutionPacker(score_set_type)
            handler_tp = functools.partial(handler_tp, packer=packer)
        else:
            handler_tp = SolutionHandler
        return handler_tp

    def _transform_matched_collection(self, solution_set_collection: List[SpectrumSolutionSet],
                                      *args, **kwargs) -> List[SpectrumSolutionSet]:
        '''This helper method can be used to re-write the target
        attribute of spectrm matches. By default, it is a no-op.

        Returns
        -------
        :class:`list` of :class:`SpectrumSolutionSet`
        '''
        return solution_set_collection

    def _evaluate_hit_groups_single_process(self, workload: WorkloadManager, *args, **kwargs) -> Mapping[str, List]:
        batch_size, scan_map, hit_map, hit_to_scan_map, scan_hit_type_map, hit_group_map = workload
        handler_tp = self._get_solution_handler()
        processor = SequentialIdentificationProcessor(
            self,
            self.mass_shift_map,
            evaluation_args=kwargs,
            solution_handler_type=handler_tp)
        processor.process(scan_map, hit_map, hit_to_scan_map,
                          scan_hit_type_map, hit_group_map)
        return processor.scan_solution_map

    def collect_scan_solutions(self, scan_solution_map: Mapping[str, List],
                               scan_map: Mapping[str, ProcessedScan],
                               *args, **kwags) -> List[SpectrumSolutionSet]:
        result_set = []
        for scan_id, solutions in scan_solution_map.items():
            if len(solutions) == 0:
                continue
            scan = scan_map[scan_id]
            out = self.solution_set_type(scan, solutions).sort()
            # This is necessary to reduce the degree to which junk matches need to have their targets re-built
            out.select_top()
            if len(out) == 0:
                continue
            result_set.append(out)
        self._transform_matched_collection(result_set, *args, **kwags)
        return result_set

    @property
    def _worker_specification(self):
        raise NotImplementedError()

    def _evaluate_hit_groups_multiple_processes(self, workload: WorkloadManager, **kwargs):
        batch_size, scan_map, hit_map, hit_to_scan, scan_hit_type_map, hit_group_map = workload
        worker_type, init_args = self._worker_specification
        handler_tp = self._get_solution_handler()
        dispatcher = IdentificationProcessDispatcher(
            worker_type, self.scorer_type, evaluation_args=kwargs, init_args=init_args,
            n_processes=self.n_processes, ipc_manager=self.ipc_manager,
            mass_shift_map=self.mass_shift_map, solution_handler_type=handler_tp)
        return dispatcher.process(scan_map, hit_map, hit_to_scan, scan_hit_type_map, hit_group_map)

    def evaluate_hit_groups(self, batch, **kwargs) -> Mapping[str, List]:
        if self.n_processes == 1 or len(batch.hit_map) < 2500:
            return self._evaluate_hit_groups_single_process(
                batch, **kwargs)
        else:
            return self._evaluate_hit_groups_multiple_processes(
                batch, **kwargs)

    def score_bunch(self, scans: List[ProcessedScan],
                    precursor_error_tolerance: float=1e-5, **kwargs) -> List[SpectrumSolutionSet]:
        workload = self._map_scans_to_hits(
            scans, precursor_error_tolerance)
        solutions = []
        for batch in workload.batches(self.batch_size):
            scan_solution_map = self.evaluate_hit_groups(batch, **kwargs)
            solutions += self.collect_scan_solutions(scan_solution_map, batch.scan_map)
        return solutions
