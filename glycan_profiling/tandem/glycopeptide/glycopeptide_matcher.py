from ..spectrum_matcher_base import TandemClusterEvaluatorBase, oxonium_detector
from ..chromatogram_mapping import ChromatogramMSMSMapper
from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.task import TaskBase

from .scoring import TargetDecoyAnalyzer

from .structure_loader import (
    CachingGlycopeptideParser, DecoyMakingCachingGlycopeptideParser)


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

    def filter_for_oxonium_ions(self, error_tolerance=1e-5):
        keep = []
        for scan in self.tandem_cluster:
            # ratio = oxonium_detector.gscore(scan.deconvoluted_peak_set)
            keep.append(scan)
        self.tandem_cluster = keep

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, *args, **kwargs):
        target_out = []
        decoy_out = []

        self.filter_for_oxonium_ions()
        for scan in self.tandem_cluster:
            target_result, decoy_result = self.score_one(scan, precursor_error_tolerance, *args, **kwargs)
            if len(target_result) > 0:
                target_out.append(target_result)
            if len(decoy_result) > 0:
                decoy_out.append(decoy_result)
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


class GlycopeptideDatabaseSearchIdentifier(TaskBase):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x):
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=100, limit=None, *args, **kwargs):
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
                bunch = [o.convert(fitted=False, deconvoluted=True) for o in bunch]
            self.log("... Spectra Extracted")
            t, d = TargetDecoyInterleavingGlycopeptideMatcher(
                bunch, self.scorer_type, self.structure_database).score_all(
                precursor_error_tolerance=precursor_error_tolerance,
                simplify=simplify, *args, **kwargs)
            self.log("... Spectra Searched")
            target_hits.extend(t)
            decoy_hits.extend(d)
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
        mapper = ChromatogramMSMSMapper(
            chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        mapper.distribute_orphans()
        mapper.assign_entities(threshold_fn, entity_chromatogram_type=entity_chromatogram_type)
        return mapper.chromatograms
