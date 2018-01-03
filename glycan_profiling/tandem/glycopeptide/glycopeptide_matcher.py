from collections import defaultdict, namedtuple, OrderedDict
from multiprocessing import Manager as IPCManager

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.task import TaskBase

from glycan_profiling.structure import (
    CachingGlycopeptideParser,
    DecoyMakingCachingGlycopeptideParser)

from .scoring import GroupwiseTargetDecoyAnalyzer

from ..spectrum_evaluation import TandemClusterEvaluatorBase, DEFAULT_BATCH_SIZE
from ..process_dispatcher import SpectrumIdentificationWorkerBase

from ..oxonium_ions import gscore_scanner
from ..chromatogram_mapping import ChromatogramMSMSMapper


class GlycopeptideIdentificationWorker(SpectrumIdentificationWorkerBase):
    def __init__(self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
                 spectrum_map, mass_shift_map, log_handler, parser_type):
        SpectrumIdentificationWorkerBase.__init__(
            self, input_queue, output_queue, done_event, scorer_type, evaluation_args,
            spectrum_map, mass_shift_map, log_handler=log_handler)
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
                 n_processes=5, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None, batch_size=DEFAULT_BATCH_SIZE):
        if parser_type is None:
            parser_type = self._default_parser_type()
        super(GlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts,
            batch_size=batch_size)
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
    '''Searches a single database against all spectra, where targets are
    database matches, and decoys are the reverse of the individual target
    glycopeptides.

    A spectrum has a best target match and a best decoy match tracked
    separately.

    This means that targets and decoys share the same glycan composition and
    peptide backbone mass, and ergo share stub glycopeptides. This may not produce
    "random" enough decoy matches.
    '''
    def __init__(self, tandem_cluster, scorer_type, structure_database, minimum_oxonium_ratio=0.05,
                 n_processes=5, ipc_manager=None, probing_range_for_missing_precursors=3,
                 mass_shifts=None, batch_size=DEFAULT_BATCH_SIZE):
        super(TargetDecoyInterleavingGlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, batch_size=batch_size)
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.target_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts)
        self.decoy_evaluator = DecoyGlycopeptideMatcher(
            [], self.scorer_type, self.structure_database, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts)

    def filter_for_oxonium_ions(self, error_tolerance=1e-5):
        keep = []
        for scan in self.tandem_cluster:
            ratio = gscore_scanner(scan.deconvoluted_peak_set)
            scan.oxonium_score = ratio
            if ratio >= self.minimum_oxonium_ratio:
                keep.append(scan)
            else:
                self.debug("... Skipping scan %s with G-score %f" % (scan.id, ratio))
        self.tandem_cluster = keep

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        # Map scans to target database
        workload = self.target_evaluator._map_scans_to_hits(
            scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        target_solutions = []
        workload_graph = workload.compute_workloads()
        total_work = workload.total_work_required(workload_graph)
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))
            target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            running_total_work += batch.batch_size
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(target_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            target_solutions += temp

        # Reuse mapped hits from target database using the decoy evaluator
        # since this assumes that the decoys will be direct reversals of
        # target sequences. The decoy evaluator will handle the reversals.
        decoy_solutions = []
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))

            decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(decoy_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            decoy_solutions += temp
            running_total_work += batch.batch_size
        return target_solutions, decoy_solutions

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, *args, **kwargs):
        target_out = []
        decoy_out = []

        self.filter_for_oxonium_ions()
        target_out, decoy_out = self.score_bunch(
            self.tandem_cluster, precursor_error_tolerance,
            simplify=simplify, *args, **kwargs)
        if simplify:
            for case in target_out:
                case.simplify()
                case.select_top()
            for case in decoy_out:
                case.simplify()
                case.select_top()
        target_out = [x for x in target_out if len(x) > 0]
        decoy_out = [x for x in decoy_out if len(x) > 0]
        return target_out, decoy_out


class CompetativeTargetDecoyInterleavingGlycopeptideMatcher(TargetDecoyInterleavingGlycopeptideMatcher):
    '''A variation of :class:`TargetDecoyInterleavingGlycopeptideMatcher` where
    a spectrum can have only one match which is either a target or a decoy.
    '''
    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        target_solutions, decoy_solutions = super(
            CompetativeTargetDecoyInterleavingGlycopeptideMatcher, self).score_bunch(
            scans, precursor_error_tolerance, simplify, *args, **kwargs)
        target_solutions = OrderedDict([(s.scan.id, s) for s in target_solutions])
        decoy_solutions = OrderedDict([(s.scan.id, s) for s in target_solutions])

        remove_target = []
        for key in target_solutions:
            try:
                if target_solutions[key].score > decoy_solutions[key].score:
                    decoy_solutions.pop(key)
                else:
                    remove_target.append(key)
            except KeyError:
                pass
        for key in remove_target:
            target_solutions.pop(key)
        return list(target_solutions.values()), list(decoy_solutions.values())


class ComparisonGlycopeptideMatcher(TargetDecoyInterleavingGlycopeptideMatcher):
    def __init__(self, tandem_cluster, scorer_type, target_structure_database, decoy_structure_database,
                 minimum_oxonium_ratio=0.05, n_processes=5, ipc_manager=None,
                 probing_range_for_missing_precursors=3, mass_shifts=None,
                 batch_size=DEFAULT_BATCH_SIZE):
        super(TargetDecoyInterleavingGlycopeptideMatcher, self).__init__(
            tandem_cluster, scorer_type, target_structure_database,
            verbose=False, n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts, batch_size=batch_size)
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.target_structure_database = target_structure_database
        self.decoy_structure_database = decoy_structure_database
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.target_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.target_structure_database,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts)
        self.decoy_evaluator = GlycopeptideMatcher(
            [], self.scorer_type, self.decoy_structure_database,
            n_processes=n_processes, ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            mass_shifts=mass_shifts)

    def score_bunch(self, scans, precursor_error_tolerance=1e-5, simplify=True, *args, **kwargs):
        # Map scans to target database
        workload = self.target_evaluator._map_scans_to_hits(
            scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        target_solutions = []
        workload_graph = workload.compute_workloads()
        total_work = workload.total_work_required(workload_graph)
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            running_total_work += batch.batch_size
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))

            target_scan_solution_map = self.target_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce target solutions
            temp = self._collect_scan_solutions(target_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            target_solutions += temp

        workload = self.decoy_evaluator._map_scans_to_hits(
            scans, precursor_error_tolerance)
        # Evaluate mapped target hits
        decoy_solutions = []
        workload_graph = workload.compute_workloads()
        total_work = workload.total_work_required(workload_graph)
        running_total_work = 0
        for i, batch in enumerate(workload.batches(self.batch_size)):
            running_total_work += batch.batch_size
            self.log("... Batch %d (%d/%d) %0.2f%%" % (
                i + 1, running_total_work, total_work,
                (100. * running_total_work) / total_work))

            decoy_scan_solution_map = self.decoy_evaluator._evaluate_hit_groups(
                batch, *args, **kwargs)
            # Aggregate and reduce decoy solutions
            temp = self._collect_scan_solutions(decoy_scan_solution_map, batch.scan_map)
            if simplify:
                temp = [case for case in temp if len(case) > 0]
                for case in temp:
                    try:
                        case.simplify()
                        case.select_top()
                    except IndexError:
                        self.log("Failed to simplify %r" % (case.scan.id,))
                        raise
            else:
                temp = [case for case in temp if len(case) > 0]
            decoy_solutions += temp
        return target_solutions, decoy_solutions


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
    return "%s:%0.3f:(%0.3f) ->\n%s" % (
        spectrum_solution.scan.id,
        spectrum_solution.scan.precursor_information.neutral_mass,
        spectrum_solution.best_solution().score,
        spectrum_solution.best_solution().target)


def format_identification_batch(group, n):
    representers = dict()
    group = sorted(group, key=lambda x: x.score, reverse=True)
    for ident in group:
        key = str(ident.best_solution().target)
        if key in representers:
            continue
        else:
            representers[key] = ident
    to_represent = sorted(
        representers.values(), key=lambda x: x.score, reverse=True)
    return '\n'.join(map(format_identification, to_represent[:n]))


def format_work_batch(bunch, count, total):
    ratio = "%d/%d (%0.3f%%)" % (count, total, (count * 100. / total))
    info = bunch[0].precursor_information
    if hasattr(info.precursor, "scan_id"):
        name = info.precursor.scan_id
    else:
        name = info.precursor.id
    if isinstance(info.charge, (int, float)):
        batch_header = "%s: %f (%s%r)" % (
            name, info.neutral_mass, "+" if info.charge > 0 else "-", abs(
                info.charge))
    else:
        batch_header = "%s: %f (%s)" % (
            name, info.neutral_mass, "?")
    return "Begin Batch", batch_header, ratio


class GlycopeptideDatabaseSearchIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x,
                 minimum_oxonium_ratio=0.05, scan_transformer=lambda x: x, adducts=None,
                 n_processes=5):
        if adducts is None:
            adducts = []
        if Unmodified not in adducts:
            adducts = [Unmodified] + adducts
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.scan_transformer = scan_transformer
        self.n_processes = n_processes
        self.ipc_manager = IPCManager()
        self.adducts = adducts

    def _make_evaluator(self, bunch):
        evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
            bunch, self.scorer_type, self.structure_database,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            n_processes=self.n_processes,
            ipc_manager=self.ipc_manager,
            mass_shifts=self.adducts)
        return evaluator

    def prepare_scan_set(self, scan_set):
        if hasattr(scan_set[0], 'convert'):
            out = []
            # Account for cases where the scan may be mentioned in the index, but
            # not actually present in the MS data
            for o in scan_set:
                try:
                    scan = (self.scorer_type.load_peaks(o))
                    if len(scan.deconvoluted_peak_set) > 0:
                        out.append(scan)
                except KeyError:
                    self.log("Missing Scan: %s" % (o.id,))
            scan_set = out
        out = []
        unconfirmed_precursors = []
        for scan in scan_set:
            try:
                scan.deconvoluted_peak_set = self.scan_transformer(
                    scan.deconvoluted_peak_set)
                if len(scan.deconvoluted_peak_set) > 0:
                    if scan.precursor_information.defaulted:
                        unconfirmed_precursors.append(scan)
                    else:
                        out.append(scan)
            except AttributeError:
                self.log("Missing Scan: %s" % (scan.id,))
                continue
        return out, unconfirmed_precursors

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=500, limit=None, *args, **kwargs):
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
            scan_collection, unconfirmed_precursors = self.prepare_scan_set(scan_collection)
            self.log("... %d Unconfirmed Precursor Spectra" % (len(unconfirmed_precursors,)))
            self.log("... Spectra Extracted")
            # TODO: handle unconfirmed_precursors differently here
            evaluator = self._make_evaluator(scan_collection + unconfirmed_precursors)
            t, d = evaluator.score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(o for o in t if o.score > 0.5)
            decoy_hits.extend(o for o in d if o.score > 0.5)
            t = sorted(t, key=lambda x: x.score, reverse=True)
            self.log("......\n%s" % (format_identification_batch(t, 10)))
            if count >= limit:
                self.log("Reached Limit. Halting.")
                break
        self.log('Search Done')
        return target_hits, decoy_hits

    def target_decoy(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        self.log("Running Target Decoy Analysis with %d targets and %d decoys" % (
            len(target_hits), len(decoy_hits)))
        tda = GroupwiseTargetDecoyAnalyzer(
            target_hits, decoy_hits, *args, with_pit=with_pit, **kwargs)
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
