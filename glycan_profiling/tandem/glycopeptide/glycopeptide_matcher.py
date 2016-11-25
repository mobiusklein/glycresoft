from multiprocessing import Process, Queue, Event

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.task import TaskBase

from glycan_profiling.serialize import (
    DatabaseBoundOperation, DatabaseScanDeserializer, func,
    MSScan, PrecursorInformation)

from glycan_profiling.database.disk_backed_database import (
    GlycopeptideDiskBackedStructureDatabase)

from .scoring import TargetDecoyAnalyzer
from glycan_profiling.database.structure_loader import (
    CachingGlycopeptideParser, DecoyMakingCachingGlycopeptideParser)

from ..spectrum_matcher_base import TandemClusterEvaluatorBase, gscore_scanner
from ..chromatogram_mapping import ChromatogramMSMSMapper


class GlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, parser_type=None):
        if parser_type is None:
            parser_type = self._default_parser_type()
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
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
    def cache_churn(self):
        return self.parser.churn


class DecoyGlycopeptideMatcher(GlycopeptideMatcher):
    def _default_parser_type(self):
        return DecoyMakingCachingGlycopeptideParser


class TargetDecoyInterleavingGlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database):
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.target_evaluator = GlycopeptideMatcher([], self.scorer_type, self.structure_database)
        self.decoy_evaluator = DecoyGlycopeptideMatcher([], self.scorer_type, self.structure_database)

    @property
    def cache_churn(self):
        return self.target_evaluator.cache_churn

    def filter_for_oxonium_ions(self, error_tolerance=1e-5):
        keep = []
        for scan in self.tandem_cluster:
            ratio = gscore_scanner.gscore(scan.deconvoluted_peak_set)
            scan.oxonium_score = ratio
            if ratio >= 0.1:
                keep.append(scan)
        self.tandem_cluster = keep

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, *args, **kwargs):
        scan_map, hit_map, hit_to_scan = self.target_evaluator._map_scans_to_hits(scans, precursor_error_tolerance)
        target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
        target_solutions = self._collect_scan_solutions(target_scan_solution_map, scan_map)
        decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
            scan_map, hit_map, hit_to_scan, *args, **kwargs)
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


def chunkiter(collection, size=200):
    i = 0
    while collection[i:(i + size)]:
        yield collection[i:(i + size)]
        i += size


class GlycopeptideIdentifyingProcess(Process):
    def __init__(self, database_connection_recipe, sample_run_id, hypothesis_id, scorer_type, input_queue,
                 output_queue, done_event, *args, **kwargs):
        Process.__init__(self)
        self.database_connection_recipe = database_connection_recipe
        self.sample_run_id = sample_run_id
        self.hypothesis_id = hypothesis_id
        self.scorer_type = scorer_type
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.done_event = done_event
        self.args = args
        self.kwargs = kwargs
        self.session = None
        self._connection = None
        self.structure_database = None

    def _setup_connection(self):
        self._connection = DatabaseBoundOperation(self.database_connection_recipe)
        self.session = self._connection.session
        self.structure_database = GlycopeptideDiskBackedStructureDatabase(
            self.database_connection_recipe, hypothesis_id=self.hypothesis_id)

    def _load_scan(self, scan_id):
        scan = self.session.query(MSScan).get(scan_id)
        return self.scorer_type.load_peaks(scan)

    def search(self, scans, precursor_error_tolerance=1e-5, simplify=True, chunk_size=250, *args, **kwargs):
        scans = [self._load_scan(i) for i in scans]
        evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
            scans, self.scorer_type, self.structure_database)
        targets, decoys = evaluator.score_all(
            precursor_error_tolerance=precursor_error_tolerance,
            simplify=simplify, *args, **kwargs)
        return targets, decoys

    def run(self):
        self._setup_connection()

        has_work = True
        while has_work:
            try:
                work_set = self.input_queue.get(10)
            except:
                if self.done_event.is_set():
                    has_work = False
                continue
            targets, decoys = self.search(work_set, *self.args, **self.kwargs)
            self.output_queue.put((targets, decoys))
        print(self, "Done")


class GlycopeptideDatabaseSearchIdentifierBound(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection, scorer_type, sample_run_id, hypothesis_id):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.scorer_type = scorer_type
        self.sample_run_id = sample_run_id
        self.hypothesis_id = hypothesis_id
        self.scan_loader = DatabaseScanDeserializer(database_connection, sample_run_id=sample_run_id)

    def scan_id_to_rt(self, scan_id):
        return self.scan_loader.convert_scan_id_to_retention_time(scan_id)

    def iter_scan_ids(self, chunk_size=250):
        q = self.session.query(PrecursorInformation.product_id).filter(
            PrecursorInformation.sample_run_id == self.sample_run_id).order_by(
            PrecursorInformation.neutral_mass.desc()).all()
        q = [i[0] for i in q]
        i = 0
        n = len(q)
        while i < n:
            yield q[i:(i + chunk_size)]
            i += chunk_size

    def _load_scan(self, scan_id):
        scan = self.session.query(MSScan).get(scan_id)
        return self.scorer_type.load_peaks(scan)

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=250, *args, **kwargs):
        target_hits = []
        decoy_hits = []
        total = self.session.query(func.count(MSScan.id)).filter(
            MSScan.ms_level == 2, MSScan.sample_run_id == self.sample_run_id).scalar()
        count = 0
        for scans in self.iter_scan_ids(chunk_size):
            scans = [self._load_scan(i) for i in scans]
            self.log("... Searching %s (%d/%d)" % (scans[0].precursor_information, count, total))
            evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
                scans, self.scorer_type, self.structure_database)
            t, d = evaluator.score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(t)
            decoy_hits.extend(d)
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
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        mapper.distribute_orphans()
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms


class MultipleProcessGlycopeptideDatabaseSearchIdentifierBound(GlycopeptideDatabaseSearchIdentifierBound):
    def __init__(self, database_connection, scorer_type, sample_run_id, hypothesis_id, n_processes=4):
        super(MultipleProcessGlycopeptideDatabaseSearchIdentifierBound, self).__init__(
            database_connection, scorer_type, sample_run_id, hypothesis_id)
        self.n_processes = n_processes

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=250, *args, **kwargs):
        pass


class GlycopeptideDatabaseSearchIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x):
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=750, limit=None, *args, **kwargs):
        target_hits = []
        decoy_hits = []
        total = len(self.tandem_scans)
        count = 0
        if limit is None:
            limit = float('inf')
        for bunch in chunkiter(self.tandem_scans, chunk_size):
            count += len(bunch)
            self.log("... Searching %s (%d/%d)" % (bunch[0].precursor_information, count, total))
            if hasattr(bunch[0], 'convert'):
                bunch = [self.scorer_type.load_peaks(o) for o in bunch]
            self.log("... Spectra Extracted")
            evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
                bunch, self.scorer_type, self.structure_database)
            t, d = evaluator.score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(o for o in t if o.score > 0)
            decoy_hits.extend(o for o in d if o.score > 0)
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
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        mapper.distribute_orphans()
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms
