from multiprocessing import Manager as IPCManager

from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.task import TaskBase


from .temp_store import TempFileManager, SpectrumMatchStore

from .spectrum_evaluation import TandemClusterEvaluatorBase, DEFAULT_BATCH_SIZE


class TandemDatabaseMatcherBase(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, parser_type=None,
                 n_processes=5, ipc_manager=None, probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True, mass_shifts=None, batch_size=DEFAULT_BATCH_SIZE):
        if parser_type is None:
            parser_type = self._default_parser_type()
        super(TandemDatabaseMatcherBase, self).__init__(
            tandem_cluster, scorer_type, structure_database, verbose=False, n_processes=n_processes,
            ipc_manager=ipc_manager,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            trust_precursor_fits=trust_precursor_fits,
            mass_shifts=mass_shifts,
            batch_size=batch_size)
        self.parser_type = parser_type
        self.parser = parser_type()

    def _default_parser_type(self):
        raise NotImplementedError()

    def reset_parser(self):
        self.parser = self.parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher

    @property
    def _worker_specification(self):
        raise NotImplementedError()


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
        str(spectrum_solution.best_solution().target))


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


class DatabaseSearchIdentifierBase(TaskBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x,
                 scan_transformer=lambda x: x, mass_shifts=None, n_processes=5, file_manager=None,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True):
        if file_manager is None:
            file_manager = TempFileManager()
        if mass_shifts is None:
            mass_shifts = []
        if Unmodified not in mass_shifts:
            mass_shifts = [Unmodified] + mass_shifts
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt
        self.mass_shifts = mass_shifts
        self.scan_transformer = scan_transformer

        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits

        self.n_processes = n_processes
        self.ipc_manager = IPCManager()

        self.file_manager = file_manager
        self.spectrum_match_store = SpectrumMatchStore(self.file_manager)

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

    def format_work_batch(self, bunch, count, total):
        ratio = "%d/%d (%0.3f%%)" % (count, total, (count * 100. / total))
        info = bunch[0].precursor_information
        try:
            try:
                precursor = info.precursor
                if hasattr(precursor, "scan_id"):
                    name = precursor.scan_id
                else:
                    name = precursor.id
            except (KeyError, AttributeError):
                if hasattr(bunch[0], "scan_id"):
                    name = bunch[0].scan_id
                else:
                    name = bunch[0].id
        except Exception:
            name = ""

        if isinstance(info.charge, (int, float)):
            batch_header = "%s: %f (%s%r)" % (
                name, info.neutral_mass, "+" if info.charge > 0 else "-", abs(
                    info.charge))
        else:
            batch_header = "%s: %f (%s)" % (
                name, info.neutral_mass, "?")
        return "Begin Batch", batch_header, ratio

    def _make_evaluator(self, bunch):
        raise NotImplementedError()

    def _before_search(self, *args, **kwargs):
        pass

    def _clear_database_cache(self):
        self.structure_database.clear_cache()

    def search(self, precursor_error_tolerance=1e-5, simplify=True, batch_size=500, limit=None, *args, **kwargs):
        target_hits = self.spectrum_match_store.writer("targets")
        decoy_hits = self.spectrum_match_store.writer("decoys")

        total = len(self.tandem_scans)
        count = 0

        if limit is None:
            limit = float('inf')

        self._before_search(*args, **kwargs)

        self.log("Writing Matches To %r" % (self.file_manager,))
        for scan_collection in chunkiter(self.tandem_scans, batch_size):
            count += len(scan_collection)
            for item in self.format_work_batch(scan_collection, count, total):
                self.log("... %s" % item)
            scan_collection, unconfirmed_precursors = self.prepare_scan_set(scan_collection)
            self.log("... %d Unconfirmed Precursor Spectra" % (len(unconfirmed_precursors,)))
            self.log("... Spectra Extracted")
            # TODO: handle unconfirmed_precursors differently here?
            evaluator = self._make_evaluator(scan_collection + unconfirmed_precursors)
            t, d = evaluator.score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(o for o in t if o.score > 0.5)
            decoy_hits.extend(o for o in d if o.score > 0.5)
            t = sorted(t, key=lambda x: x.score, reverse=True)
            self.log("...... Total Matches So Far: %d Targets, %d Decoys\n%s" % (
                len(target_hits), len(decoy_hits), format_identification_batch(t, 10)))
            if count >= limit:
                self.log("Reached Limit. Halting.")
                break
            # clear these lists as they may be quite large and we don't need them around for the
            # next iteration
            t = []
            d = []

        self.log('Search Done')
        target_hits.close()
        decoy_hits.close()
        self._clear_database_cache()

        self.log("Reloading Spectrum Matches")
        target_hits, decoy_hits = self._load_stored_matches(len(target_hits), len(decoy_hits))
        return target_hits, decoy_hits

    def _load_stored_matches(self, *args, **kwargs):
        raise NotImplementedError()

    def target_decoy(self, *args, **kwargs):
        raise NotImplementedError()


class DatabaseSearchComparerBase(DatabaseSearchIdentifierBase):
    def __init__(self, tandem_scans, scorer_type, target_database, decoy_database, scan_id_to_rt=lambda x: x,
                 scan_transformer=lambda x: x, mass_shifts=None, n_processes=5, file_manager=None,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True):
        self.target_database = target_database
        self.decoy_database = decoy_database
        super(DatabaseSearchComparerBase, self).__init__(
            tandem_scans, scorer_type, self.target_database, scan_id_to_rt,
            scan_transformer, mass_shifts, n_processes, file_manager, probing_range_for_missing_precursors,
            trust_precursor_fits)

    def _clear_database_cache(self):
        self.target_database.clear_cache()
        self.decoy_database.clear_cache()


class ExclusiveDatabaseSearchComparerBase(DatabaseSearchComparerBase):
    def target_decoy(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        accepted_targets, accepted_decoys = self._find_best_match_for_each_scan(target_hits, decoy_hits)
        tda = super(ExclusiveDatabaseSearchComparerBase, self).target_decoy(
            accepted_targets, accepted_decoys, with_pit=with_pit, *args, **kwargs)
        for sol in target_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        for sol in decoy_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        return tda

    def _find_best_match_for_each_scan(self, target_hits, decoy_hits):
        winning_targets = []
        winning_decoys = []

        target_map = {t.scan.id: t for t in target_hits}
        decoy_map = {t.scan.id: t for t in decoy_hits}
        scan_ids = set(target_map) | set(decoy_map)
        for scan_id in scan_ids:
            target_sol = target_map.get(scan_id)
            decoy_sol = decoy_map.get(scan_id)
            if target_sol is None:
                winning_decoys.append(decoy_sol)
            elif decoy_sol is None:
                winning_targets.append(target_sol)
            else:
                if target_sol.score == decoy_sol.score:
                    winning_targets.append(target_sol)
                    winning_decoys.append(decoy_sol)
                elif target_sol.score > decoy_sol.score:
                    winning_targets.append(target_sol)
                else:
                    winning_decoys.append(decoy_sol)
        return winning_targets, winning_decoys
