from multiprocessing import Manager as IPCManager

from glycresoft.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycresoft.chromatogram_tree import Unmodified
from glycresoft.task import TaskBase

from glycresoft.structure import (
    CachingGlycopeptideParser,
    SequenceReversingCachingGlycopeptideParser)


from ..target_decoy import GroupwiseTargetDecoyAnalyzer, TargetDecoySet
from .core_search import GlycanCombinationRecord, GlycanFilteringPeptideMassEstimator

from ..temp_store import TempFileManager, SpectrumMatchStore
from ..chromatogram_mapping import ChromatogramMSMSMapper
from ..workflow import (format_identification_batch, chunkiter, SearchEngineBase)

from ..oxonium_ions import OxoniumFilterReport, OxoniumFilterState

from .matcher import (
    TargetDecoyInterleavingGlycopeptideMatcher,
    ComparisonGlycopeptideMatcher)


class GlycopeptideResolver(object):
    def __init__(self, database, parser=None):
        if parser is None:
            parser = CachingGlycopeptideParser()
        self.database = database
        self.parser = parser
        self.cache = dict()

    def resolve(self, id):
        try:
            return self.cache[id]
        except KeyError:
            record = self.database.get_record(id)
            structure = self.parser(record)
            self.cache[id] = structure
            return structure

    def __call__(self, id):
        return self.resolve(id)


class GlycopeptideDatabaseSearchIdentifier(SearchEngineBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x,
                 minimum_oxonium_ratio=0.05, scan_transformer=lambda x: x, mass_shifts=None,
                 n_processes=5, file_manager=None, use_peptide_mass_filter=True,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 permute_decoy_glycans=False):
        if file_manager is None:
            file_manager = TempFileManager()
        elif isinstance(file_manager, str):
            file_manager = TempFileManager(file_manager)
        if mass_shifts is None:
            mass_shifts = []
        if Unmodified not in mass_shifts:
            mass_shifts = [Unmodified] + mass_shifts
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt
        self.minimum_oxonium_ratio = minimum_oxonium_ratio
        self.mass_shifts = mass_shifts
        self.permute_decoy_glycans = permute_decoy_glycans

        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits

        self.use_peptide_mass_filter = use_peptide_mass_filter
        self._peptide_mass_filter = None

        self.scan_transformer = scan_transformer

        self.n_processes = n_processes
        self.ipc_manager = IPCManager()

        self.file_manager = file_manager
        self.spectrum_match_store = SpectrumMatchStore(self.file_manager)

    def _make_evaluator(self, bunch):
        evaluator = TargetDecoyInterleavingGlycopeptideMatcher(
            bunch, self.scorer_type, self.structure_database,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            n_processes=self.n_processes,
            ipc_manager=self.ipc_manager,
            mass_shifts=self.mass_shifts,
            peptide_mass_filter=self._peptide_mass_filter,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            permute_decoy_glycans=self.permute_decoy_glycans)
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

    def _make_peptide_mass_filter(self, error_tolerance=1e-5):
        hypothesis_id = self.structure_database.hypothesis_id
        glycan_combination_list = GlycanCombinationRecord.from_hypothesis(
            self.structure_database.session, hypothesis_id)
        if len(glycan_combination_list) == 0:
            self.log("No glycan combinations were found")
            raise ValueError("No glycan combinations were found")
        peptide_filter = GlycanFilteringPeptideMassEstimator(
            glycan_combination_list, product_error_tolerance=error_tolerance)
        return peptide_filter

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

    def search(self, precursor_error_tolerance=1e-5, simplify=True, batch_size=500, *args, **kwargs):
        target_hits = self.spectrum_match_store.writer("targets")
        decoy_hits = self.spectrum_match_store.writer("decoys")

        total = len(self.tandem_scans)
        count = 0

        if self.use_peptide_mass_filter:
            self._peptide_mass_filter = self._make_peptide_mass_filter(
                kwargs.get("error_tolerance", 1e-5))
        oxonium_report = OxoniumFilterReport()
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

            # clear these lists as they may be quite large and we don't need them around for the
            # next iteration
            t = []
            d = []
            oxonium_report.extend(evaluator.oxonium_ion_report)

        self.log('Search Done')
        target_hits.close()
        decoy_hits.close()
        self._clear_database_cache()

        self.log("Reloading Spectrum Matches")
        target_hits, decoy_hits = self._load_stored_matches(len(target_hits), len(decoy_hits))
        return TargetDecoySet(target_hits, decoy_hits)

    def _clear_database_cache(self):
        self.structure_database.clear_cache()

    def _load_stored_matches(self, target_count, decoy_count):
        target_resolver = GlycopeptideResolver(self.structure_database, CachingGlycopeptideParser(int(1e6)))
        decoy_resolver = GlycopeptideResolver(self.structure_database, SequenceReversingCachingGlycopeptideParser(int(1e6)))

        loaded_target_hits = []
        for i, solset in enumerate(self.spectrum_match_store.reader("targets", target_resolver)):
            if i % 5000 == 0:
                self.log("Loaded %d/%d Targets (%0.3g%%)" % (i, target_count, (100. * i / target_count)))
            loaded_target_hits.append(solset)
        loaded_decoy_hits = []
        for i, solset in enumerate(self.spectrum_match_store.reader("decoys", decoy_resolver)):
            if i % 5000 == 0:
                self.log("Loaded %d/%d Decoys (%0.3g%%)" % (i, decoy_count, (100. * i / decoy_count)))
            loaded_decoy_hits.append(solset)
        return TargetDecoySet(loaded_target_hits, loaded_decoy_hits)

    def estimate_fdr(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        self.log("Running Target Decoy Analysis with %d targets and %d decoys" % (
            len(target_hits), len(decoy_hits)))

        def over_10_aa(x):
            return len(x.target) >= 10

        def under_10_aa(x):
            return len(x.target) < 10

        grouping_fns = [over_10_aa, under_10_aa]

        tda = GroupwiseTargetDecoyAnalyzer(
            [x.best_solution() for x in target_hits],
            [x.best_solution() for x in decoy_hits], *args, with_pit=with_pit,
            grouping_functions=grouping_fns,
            grouping_labels=["Long Peptide", "Short Peptide"],
            **kwargs)
        tda.q_values()
        for sol in target_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        for sol in decoy_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        return tda

    def map_to_chromatograms(self, chromatograms, tandem_identifications,
                             precursor_error_tolerance=1e-5, threshold_fn=lambda x: x.q_value < 0.05,
                             entity_chromatogram_type=GlycopeptideChromatogram):
        self.log("Mapping MS/MS Identifications onto Chromatograms")
        self.log("%d Chromatograms" % len(chromatograms))
        # if len(chromatograms) == 0:
        #     self.log("No Chromatograms Extracted!")
        #     return chromatograms, tandem_identifications
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        self.log("Assigning Solutions")
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        self.log("Distributing Orphan Spectrum Matches")
        mapper.distribute_orphans(threshold_fn=threshold_fn)
        self.log("Selecting Most Representative Matches")
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms, mapper.orphans


class GlycopeptideDatabaseSearchComparer(GlycopeptideDatabaseSearchIdentifier):
    def __init__(self, tandem_scans, scorer_type, target_database, decoy_database, scan_id_to_rt=lambda x: x,
                 minimum_oxonium_ratio=0.05, scan_transformer=lambda x: x, mass_shifts=None,
                 n_processes=5, file_manager=None, use_peptide_mass_filter=True,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 permute_decoy_glycans=False):
        self.target_database = target_database
        self.decoy_database = decoy_database
        super(GlycopeptideDatabaseSearchComparer, self).__init__(
            tandem_scans, scorer_type, self.target_database, scan_id_to_rt,
            minimum_oxonium_ratio, scan_transformer, mass_shifts, n_processes,
            file_manager, use_peptide_mass_filter,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            trust_precursor_fits=trust_precursor_fits, permute_decoy_glycans=permute_decoy_glycans)

    def _clear_database_cache(self):
        self.target_database.clear_cache()
        self.decoy_database.clear_cache()

    def _make_evaluator(self, bunch):
        evaluator = ComparisonGlycopeptideMatcher(
            bunch, self.scorer_type,
            target_structure_database=self.target_database,
            decoy_structure_database=self.decoy_database,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            n_processes=self.n_processes,
            ipc_manager=self.ipc_manager,
            mass_shifts=self.mass_shifts,
            peptide_mass_filter=self._peptide_mass_filter,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            permute_decoy_glycans=self.permute_decoy_glycans)
        return evaluator

    def estimate_fdr(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        self.log("Running Target Decoy Analysis with %d targets and %d decoys" % (
            len(target_hits), len(decoy_hits)))

        database_ratio = float(len(self.target_database)) / len(self.decoy_database)

        def over_10_aa(x):
            return len(x.target) >= 10

        def under_10_aa(x):
            return len(x.target) < 10

        grouping_fns = [over_10_aa, under_10_aa]

        tda = GroupwiseTargetDecoyAnalyzer(
            [x.best_solution() for x in target_hits],
            [x.best_solution() for x in decoy_hits], *args, with_pit=with_pit,
            database_ratio=database_ratio, grouping_functions=grouping_fns,
            grouping_labels=["Long Peptide", "Short Peptide"]
            **kwargs)

        tda.q_values()
        for sol in target_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        for sol in decoy_hits:
            for hit in sol:
                tda.score(hit)
            sol.q_value = sol.best_solution().q_value
        return tda

    def _load_stored_matches(self, target_count, decoy_count):
        target_resolver = GlycopeptideResolver(self.target_database, CachingGlycopeptideParser(int(1e6)))
        decoy_resolver = GlycopeptideResolver(self.decoy_database, CachingGlycopeptideParser(int(1e6)))

        loaded_target_hits = []
        for i, solset in enumerate(self.spectrum_match_store.reader("targets", target_resolver)):
            if i % 5000 == 0:
                self.log("Loaded %d/%d Targets (%0.3g%%)" % (i, target_count, (100. * i / target_count)))
            loaded_target_hits.append(solset)
        loaded_decoy_hits = []
        for i, solset in enumerate(self.spectrum_match_store.reader("decoys", decoy_resolver)):
            if i % 5000 == 0:
                self.log("Loaded %d/%d Decoys (%0.3g%%)" % (i, decoy_count, (100. * i / decoy_count)))
            loaded_decoy_hits.append(solset)
        return TargetDecoySet(loaded_target_hits, loaded_decoy_hits)


class ExclusiveGlycopeptideDatabaseSearchComparer(GlycopeptideDatabaseSearchComparer):
    def estimate_fdr(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        accepted_targets, accepted_decoys = self._find_best_match_for_each_scan(target_hits, decoy_hits)
        tda = super(ExclusiveGlycopeptideDatabaseSearchComparer, self).estimate_fdr(
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
