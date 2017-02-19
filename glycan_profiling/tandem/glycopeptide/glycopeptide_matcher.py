from collections import defaultdict, namedtuple

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.task import TaskBase

from glycan_profiling.serialize import (
    DatabaseBoundOperation, DatabaseScanDeserializer, func,
    MSScan, PrecursorInformation)

from glycan_profiling.database.disk_backed_database import (
    GlycopeptideDiskBackedStructureDatabase)
from glycan_profiling.database.mass_collection import ConcatenatedDatabase

from .scoring import TargetDecoyAnalyzer
from glycan_profiling.database.structure_loader import (
    CachingGlycopeptideParser, DecoyMakingCachingGlycopeptideParser)

from ..spectrum_matcher_base import (
    TandemClusterEvaluatorBase, gscore_scanner,
    IdentificationProcessDispatcher,
    SpectrumIdentificationWorkerBase, Manager as IPCManager)
from ..chromatogram_mapping import ChromatogramMSMSMapper


class GlycopeptideIdentificationWorker(SpectrumIdentificationWorkerBase):
    def __init__(self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
                 spectrum_map, log_handler, parser_type):
        SpectrumIdentificationWorkerBase.__init__(
            self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
            spectrum_map, log_handler=log_handler)
        self.parser = parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher


_target_decoy_cell = namedtuple("_target_decoy_cell", ["target", "decoy"])


def make_target_decoy_cell():
    return _target_decoy_cell(target=None, decoy=None)


class GlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, parser_type=None,
                 n_processes=5, ipc_manager=None):
        if parser_type is None:
            parser_type = self._default_parser_type()
        super(GlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False, n_processes=n_processes,
            ipc_manager=ipc_manager)
        self.parser_type = parser_type
        self.parser = parser_type()

    def _default_parser_type(self):
        return CachingGlycopeptideParser

    def reset_parser(self):
        self.parser = self.parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher

    @property
    def _worker_specification(self):
        return GlycopeptideIdentificationWorker, {"parser_type": self.parser_type}


class DecoyGlycopeptideMatcher(GlycopeptideMatcher):
    def _default_parser_type(self):
        return DecoyMakingCachingGlycopeptideParser


class TargetDecoyInterleavingGlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, minimum_oxonium_ratio=0.05,
                 n_processes=5, ipc_manager=None):
        super(TargetDecoyInterleavingGlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False,
            n_processes=n_processes, ipc_manager=ipc_manager)
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.target_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager)
        self.decoy_evaluator = DecoyGlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager)

    def filter_for_oxonium_ions(self, error_tolerance=1e-5):
        keep = []
        for scan in self.tandem_cluster:
            ratio = gscore_scanner(scan.deconvoluted_peak_set)
            scan.oxonium_score = ratio
            if ratio >= self.minimum_oxonium_ratio:
                keep.append(scan)
        self.tandem_cluster = keep

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, *args, **kwargs):
        # Map scans to target database
        scan_map, hit_map, hit_to_scan = self.target_evaluator._map_scans_to_hits(scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
        # Aggregate and reduce target solutions
        target_solutions = self._collect_scan_solutions(target_scan_solution_map, scan_map)

        # Reuse mapped hits from target database using the decoy evaluator
        # since this assumes that the decoys will be direct reversals of
        # target sequences. The decoy evaluator will handle the reversals.
        decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
        # Aggregate and reduce target solutions
        decoy_solutions = self._collect_scan_solutions(decoy_scan_solution_map, scan_map)
        return target_solutions, decoy_solutions

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, *args, **kwargs):
        target_out = []
        decoy_out = []

        self.filter_for_oxonium_ions()
        target_out, decoy_out = self.score_bunch(self.tandem_cluster, precursor_error_tolerance, *args, **kwargs)
        if simplify:
            for case in target_out:
                case.simplify()
                case.select_top()
            for case in decoy_out:
                case.simplify()
                case.select_top()
        return target_out, decoy_out


# These matchers are missing patches for parallelism
class ComparisonGlycopeptideMatcher(TargetDecoyInterleavingGlycopeptideMatcher):
    def __init__(self, tandem_cluster, scorer_type, target_structure_database, decoy_structure_database,
                 minimum_oxonium_ratio=0.05):
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.target_structure_database = target_structure_database
        self.decoy_structure_database = decoy_structure_database
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.target_evaluator = GlycopeptideMatcher([], self.scorer_type, self.target_structure_database)
        self.decoy_evaluator = GlycopeptideMatcher([], self.scorer_type, self.decoy_structure_database)

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, *args, **kwargs):
        # Map scans to target database
        scan_map, hit_map, hit_to_scan = self.target_evaluator._map_scans_to_hits(scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
        # Aggregate and reduce target solutions
        target_solutions = self._collect_scan_solutions(target_scan_solution_map, scan_map)

        # Map scans to decoy database
        scan_map, hit_map, hit_to_scan = self.decoy_evaluator._map_scans_to_hits(scans, precursor_error_tolerance)
        # Evaluate mapped decoy hits
        decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
        # Aggregate and reduce decoy solutions
        decoy_solutions = self._collect_scan_solutions(decoy_scan_solution_map, scan_map)
        return target_solutions, decoy_solutions


# These matchers are missing patches for parallelism
class ConcatenatedGlycopeptideMatcher(ComparisonGlycopeptideMatcher):
    def score_bunch(self, scans, precursor_information=1e-5, *args, **kwargs):
        target_solutions, decoy_solutions = super(ConcatenatedGlycopeptideMatcher, self).score_bunch(
            scans, precursor_information, *args, **kwargs)
        aggregator = defaultdict(make_target_decoy_cell)

        for solution in target_solutions:
            aggregator[solution.scan.id].target = solution
        for solution in decoy_solutions:
            aggregator[solution.scan.id].decoy = solution

        target_solutions = []
        decoy_solutions = []

        for scan_id, cell in aggregator.items():
            if cell.target is not None:
                target_score = cell.target.score
            else:
                target_score = 0.0
            if cell.decoy is not None:
                decoy_score = cell.decoy.score
            else:
                decoy_score = 0.0

            if target_score > decoy_score:
                target_solutions.append(cell.target)
            else:
                decoy_solutions.append(cell.decoy)

        return target_solutions, decoy_solutions


def chunkiter(collection, size=200):
    i = 0
    while collection[i:(i + size)]:
        yield collection[i:(i + size)]
        i += size


def format_identification(spectrum_solution):
    return "%s:(%0.3f) ->\n%s" % (
        spectrum_solution.scan.id,
        spectrum_solution.best_solution().score,
        spectrum_solution.best_solution().target)


def format_work_batch(bunch, count, total):
    ratio = "%d/%d (%0.3f%%)" % (count, total, (count * 100. / total))
    info = bunch[0].precursor_information
    if hasattr(info.precursor, "scan_id"):
        name = info.precursor.scan_id
    else:
        name = info.precursor.id
    batch_header = "%s: %f (%s%r)" % (
        name, info.neutral_mass, "+" if info.charge > 0 else "-", abs(
            info.charge))
    return "Begin Batch", batch_header, ratio


class GlycopeptideDatabaseSearchIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x,
                 minimum_oxonium_ratio=0.05, scan_transformer=lambda x: x, n_processes=5):
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.scan_transformer = scan_transformer
        self.n_processes = n_processes
        self.ipc_manager = IPCManager()

    def _make_evaluator(self, bunch):
        evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
            bunch, self.scorer_type, self.structure_database,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            n_processes=self.n_processes,
            ipc_manager=self.ipc_manager)
        return evaluator

    def prepare_scan_set(self, scan_set):
        if hasattr(scan_set[0], 'convert'):
            out = []
            # Account for cases where the scan may be mentioned in the index, but
            # not actually present in the MS data
            for o in scan_set:
                try:
                    out.append(self.scorer_type.load_peaks(o))
                except KeyError:
                    self.log("Missing Scan: %s" % (o.id,))
            scan_set = out
        out = []
        for scan in scan_set:
            try:
                scan.deconvoluted_peak_set = self.scan_transformer(
                    scan.deconvoluted_peak_set)
                out.append(scan)
            except AttributeError:
                self.log("Missing Scan: %s" % (scan.id,))
                continue
        return out

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=1000, limit=None, *args, **kwargs):
        target_hits = []
        decoy_hits = []
        total = len(self.tandem_scans)
        count = 0
        if limit is None:
            limit = float('inf')
        for scan_collection in chunkiter(self.tandem_scans, chunk_size):
            count += len(scan_collection)
            for item in format_work_batch(scan_collection, count, total):
                self.log("... %s" % item)
            scan_collection = self.prepare_scan_set(scan_collection)
            self.log("... Spectra Extracted")
            evaluator = self._make_evaluator(scan_collection)
            t, d = evaluator.score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(o for o in t if o.score > 0)
            decoy_hits.extend(o for o in d if o.score > 0)
            t = sorted(t, key=lambda x: x.score, reverse=True)
            self.log("......\n%s" % ('\n'.join(map(format_identification, t[:4]))))
            if count >= limit:
                self.log("Reached Limit. Halting.")
                break
        self.log('Search Done')
        return target_hits, decoy_hits

    def target_decoy(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        self.log("Running Target Decoy Analysis with %d targets and %d decoys" % (
            len(target_hits), len(decoy_hits)))
        tda = TargetDecoyAnalyzer(target_hits, decoy_hits, *args, with_pit=with_pit, **kwargs)
        tda.q_values()
        for sol in target_hits:
            for hit in sol:
                tda.score(hit)
        for sol in decoy_hits:
            for hit in sol:
                tda.score(hit)
        return tda

    def map_to_chromatograms(self, chromatograms, tandem_identifications,
                             precursor_error_tolerance=1e-5, threshold_fn=lambda x: x.q_value < 0.05,
                             entity_chromatogram_type=GlycopeptideChromatogram):
        self.log("Mapping MS/MS Identifications onto Chromatograms")
        self.log("%d Chromatograms" % len(chromatograms))
        if len(chromatograms) == 0:
            self.log("No Chromatograms Extracted!")
            return chromatograms
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        mapper.distribute_orphans()
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms, mapper.orphans


class GlycopeptideDatabaseSearchComparer(GlycopeptideDatabaseSearchIdentifier):
    def __init__(self, tandem_scans, scorer_type, target_database, decoy_database, scan_id_to_rt=lambda x: x,
                 minimum_oxonium_ratio=0.05):
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.target_database = target_database
        self.decoy_database = decoy_database
        self.scan_id_to_rt = scan_id_to_rt
        self.minimum_oxonium_ratio = minimum_oxonium_ratio

    def _make_evaluator(self, bunch):
        evaluator = ComparisonGlycopeptideMatcher(
            bunch, self.scorer_type,
            target_structure_database=self.target_database,
            decoy_structure_database=self.decoy_database,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio)
        return evaluator


class ConcatenatedGlycopeptideDatabaseSearchComparer(GlycopeptideDatabaseSearchIdentifier):
    pass
