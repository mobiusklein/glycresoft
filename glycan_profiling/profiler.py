import re
import time
import warnings
import logging

import glypy

from glycan_profiling import plotting
from glycan_profiling.database import build_database
from glycan_profiling.database.disk_backed_database import GlycanCompositionDiskBackedStructureDatabase

from glycan_profiling.serialize import (
    DatabaseScanDeserializer, AnalysisSerializer)

from glycan_profiling.piped_deconvolve import (
    ScanGenerator as PipedScanGenerator, MzMLLoader)

from glycan_profiling.scoring import (
    ChromatogramSolution, NetworkScoreDistributor, ChromatogramScorer)

from glycan_profiling.chromatogram_tree import (
    ChromatogramOverlapSmoother, GlycanCompositionChromatogram, ChromatogramFilter)

from glycan_profiling.trace import (
    IncludeUnmatchedTracer, join_mass_shifted,
    reverse_adduction_search, prune_bad_adduct_branches,
    NonAggregatingTracer, ScanSink, ChromatogramExtractor,
    ChromatogramProcessor)

from glycan_profiling.scan_cache import (
    NullScanCacheHandler, ThreadedDatabaseScanCacheHandler,
    DatabaseScanGenerator)

from glycan_profiling.task import TaskBase

from brainpy import periodic_table
from ms_deisotope.averagine import Averagine, glycan as n_glycan_averagine


logger = logging.getLogger("chromatogram_profiler")


def validate_element(element):
    valid = element in periodic_table
    if not valid:
        warnings.warn("%r is not a valid element" % element)
    return valid


def parse_averagine_formula(formula):
    return Averagine({k: float(v) for k, v in re.findall(r"([A-Z][a-z]*)([0-9\.]*)", formula)
                      if float(v or 0) > 0 and validate_element(k)})


class SampleConsumer(TaskBase):
    def __init__(self, mzml_path, averagine=n_glycan_averagine, charge_range=(-1, -8),
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, ms1_deconvolution_args=None,
                 msn_deconvolution_args=None, start_scan_id=None, end_scan_id=None, storage_path=None,
                 sample_name=None, cache_handler_type=None):

        if cache_handler_type is None:
            cache_handler_type = ThreadedDatabaseScanCacheHandler
        if isinstance(averagine, basestring):
            averagine = parse_averagine_formula(averagine)

        self.mzml_path = mzml_path
        self.storage_path = storage_path
        self.sample_name = sample_name

        self.cache_handler_type = cache_handler_type

        self.scan_generator = PipedScanGenerator(
            mzml_path, averagine=averagine, charge_range=charge_range, ms1_peak_picking_args=ms1_peak_picking_args,
            msn_peak_picking_args=msn_peak_picking_args, ms1_deconvolution_args=ms1_deconvolution_args,
            msn_deconvolution_args=msn_deconvolution_args)

        self.start_scan_id = start_scan_id
        self.end_scan_id = end_scan_id

    def run(self):
        self.scan_generator.configure_iteration(self.start_scan_id, self.end_scan_id)

        sink = ScanSink(self.scan_generator, self.cache_handler_type)
        sink.configure_cache(self.storage_path, self.sample_name)

        for scan in sink:
            self.log("Processed %s (%f)" % (scan.id, scan.scan_time))

        sink.complete()
        self.log("Completed Sample %s" % (self.sample_name,))
        sink.commit()


class GlycanChromatogramAnalyzer(TaskBase):
    def __init__(self, database_connection, hypothesis_id, sample_run_id, adducts=None,
                 mass_error_tolerance=1e-5, grouping_error_tolerance=1.5e-5,
                 scoring_model=None, analysis_name=None):
        self.database_connection = database_connection
        self.hypothesis_id = hypothesis_id
        self.sample_run_id = sample_run_id
        self.mass_error_tolerance = mass_error_tolerance
        self.grouping_error_tolerance = grouping_error_tolerance
        self.scoring_model = scoring_model
        self.adducts = adducts
        self.analysis_name = analysis_name

    def save_solutions(self, solutions, peak_mapping):
        analysis_saver = AnalysisSerializer(self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(peak_mapping)
        for chroma in solutions:
            if chroma.composition:
                analysis_saver.save_glycan_composition_chromatogram_solution(chroma)
            else:
                analysis_saver.save_unidentified_chromatogram_solution(chroma)
        analysis_saver.commit()

    def run(self):
        peak_loader = DatabaseScanDeserializer(
            self.database_connection, sample_run_id=self.sample_run_id)

        database = GlycanCompositionDiskBackedStructureDatabase(
            self.database_connection, self.hypothesis_id)

        extractor = ChromatogramExtractor(
            peak_loader, grouping_tolerance=self.grouping_error_tolerance)
        proc = ChromatogramProcessor(
            extractor, database, mass_error_tolerance=self.mass_error_tolerance, adducts=self.adducts,
            scoring_model=self.scoring_model)
        proc.start()
        self.save_solutions(proc.solutions, extractor.peak_mapping)
        return proc


class ChromatogramProfiler(TaskBase):
    def __init__(self, mzml_path, database_rules_path, averagine=n_glycan_averagine, charge_range=(-1, -8),
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, ms1_deconvolution_args=None,
                 msn_deconvolution_args=None, storage_path=None, sample_name=None, analysis_name=None,
                 cache_handler_type=None, aggregate=True):
        if cache_handler_type is None:
            cache_handler_type = ThreadedDatabaseScanCacheHandler
        if isinstance(averagine, basestring):
            averagine = parse_averagine_formula(averagine)

        self.mzml_path = mzml_path
        self.storage_path = storage_path
        self.sample_name = sample_name
        self.analysis_name = analysis_name

        self.cache_handler_type = cache_handler_type

        self.scan_generator = PipedScanGenerator(
            mzml_path, averagine=averagine, charge_range=charge_range, ms1_peak_picking_args=ms1_peak_picking_args,
            msn_peak_picking_args=msn_peak_picking_args, ms1_deconvolution_args=ms1_deconvolution_args,
            msn_deconvolution_args=msn_deconvolution_args)

        self.database_rules_path = database_rules_path
        if isinstance(database_rules_path, basestring):
            self.database = build_database(database_rules_path)
        else:
            self.database = database_rules_path

        self.tracer = None
        self.adducts = []

        self.chromatograms = None
        self.solutions = None
        self._aggregate = aggregate

    def search(self, mass_error_tolerance=1e-5, grouping_mass_error_tolerance=None, start_scan=None,
               max_scans=None, end_scan=None, adducts=None, truncate=False, minimum_mass=500):

        start = time.time()

        if adducts is None:
            adducts = []
        self.adducts = adducts

        logger.info("Begin Chromatogram Tracing")
        self.scan_generator.configure_iteration(
            start_scan=start_scan, end_scan=end_scan, max_scans=max_scans)
        if self._aggregate:
            self.tracer = IncludeUnmatchedTracer(
                self.scan_generator, self.database, mass_error_tolerance,
                cache_handler_type=self.cache_handler_type)
        else:
            self.tracer = NonAggregatingTracer(
                self.scan_generator, cache_handler_type=self.cache_handler_type)
        self.tracer.configure_cache(self.storage_path, self.sample_name)

        i = 0
        for case in self.tracer:
            self.log("%d, %d, %s, %s, %d" % (
                i, case[1].index, case[1].scan_time, case[1].id, len(case[0])))
            i += 1
            if end_scan == case[1].id or i == max_scans:
                break

        self.tracer.commit()
        self.tracer.complete()

        if grouping_mass_error_tolerance is None:
            grouping_mass_error_tolerance = mass_error_tolerance * 1.5

        self.build_chromatograms(
            mass_error_tolerance, grouping_mass_error_tolerance, adducts, truncate)
        self.log("Tracing Complete (%r minutes elapsed)" % ((time.time() - start) / 60.,))

    def build_chromatograms(self, mass_error_tolerance, grouping_mass_error_tolerance, adducts,
                            truncate=False, minimum_mass=500):
        self.log("Post-Processing Chromatogram Traces (%0.3e, %0.3e, %r, %r)" % (
            mass_error_tolerance, grouping_mass_error_tolerance, adducts, truncate))
        self.chromatograms = self.tracer.build_chromatograms(
            grouping_tolerance=grouping_mass_error_tolerance,
            truncate=truncate,
            minimum_mass=minimum_mass)
        self._convert_to_entity_chromatograms()
        self.chromatograms = reverse_adduction_search(
            join_mass_shifted(
                ChromatogramFilter(self.chromatograms), adducts, mass_error_tolerance),
            adducts, mass_error_tolerance, self.database)
        self._convert_to_entity_chromatograms()

    def _convert_to_entity_chromatograms(self, entity_chromatogram_type=GlycanCompositionChromatogram):
        acc = []
        for chroma in self.chromatograms:
            copied = chroma.clone(entity_chromatogram_type)
            copied.entity = copied.composition
            acc.append(copied)
        self.chromatograms = ChromatogramFilter(acc, sort=False)

    def _convert_solution_to_entity_chromatogram(self, entity_chromatogram_type=GlycanCompositionChromatogram):
        for chroma in self.solutions:
            copied = chroma.chromatogram.clone(entity_chromatogram_type)
            copied.entity = copied.composition
            chroma.chromatogram = copied

    def _evaluate_fits(self, chromatograms, base_coef=0.8, support_coef=0.2, rt_delta=0.25,
                       scoring_model=None, min_points=3, smooth=True):
        solutions = []
        if scoring_model is None:
            scoring_model = ChromatogramScorer()

        filtered = ChromatogramFilter.process(
            chromatograms, delta_rt=rt_delta, min_points=min_points)

        if smooth:
            filtered = ChromatogramOverlapSmoother(filtered)

        for case in filtered:
            try:
                solutions.append(ChromatogramSolution(case, scorer=scoring_model))
            except (IndexError, ValueError), e:
                print case, e, len(case)
                continue

        if base_coef != 1.0:
            NetworkScoreDistributor(solutions, self.database.network).distribute(base_coef, support_coef)
        return solutions

    def score(self, base_coef=0.8, support_coef=0.2, rt_delta=0.25, scoring_model=None, min_points=3,
              smooth=True):
        if scoring_model is None:
            scoring_model = ChromatogramScorer()

        self.solutions = self._evaluate_fits(
            self.chromatograms, base_coef, support_coef, rt_delta, scoring_model, min_points,
            smooth)

        if self.adducts:
            hold = prune_bad_adduct_branches(ChromatogramFilter(self.solutions))
            self.solutions = self._evaluate_fits(hold, base_coef, support_coef, rt_delta, scoring_model)

        self.solutions = ChromatogramFilter(sol for sol in self.solutions if sol.score > 1e-5)
        self._convert_solution_to_entity_chromatogram()
        return self._filter_accepted()

    def _filter_accepted(self, threshold=0.4):
        self.accepted_solutions = ChromatogramFilter([
            sol for sol in self.solutions
            if sol.score > threshold and not sol.used_as_adduct
        ])
        return self.accepted_solutions

    def plot(self, min_score=0.4, min_signal=0.2, colorizer=None, chromatogram_artist=None, include_tic=True):
        if chromatogram_artist is None:
            chromatogram_artist = plotting.SmoothingChromatogramArtist
        monosaccharides = set()

        for sol in self.solutions:
            if sol.glycan_composition:
                monosaccharides.update(map(str, sol.glycan_composition))

        label_abundant = plotting.AbundantLabeler(
            plotting.NGlycanLabelProducer(monosaccharides),
            max(sol.total_signal for sol in self.solutions if sol.score > min_score) * min_signal)

        if colorizer is None:
            colorizer = plotting.n_glycan_colorizer

        results = [sol for sol in self.solutions if sol.score > min_score and not sol.used_as_adduct]
        chrom = chromatogram_artist(results, colorizer=colorizer).draw(label_function=label_abundant)
        if include_tic:
            chrom.draw_generic_chromatogram(
                "TIC",
                map(self.tracer.scan_id_to_rt, self.tracer.total_ion_chromatogram),
                self.tracer.total_ion_chromatogram.values(), 'blue')
            chrom.ax.set_ylim(0, max(self.tracer.total_ion_chromatogram.values()) * 1.1)

        agg = plotting.AggregatedAbundanceArtist(results)
        agg.draw()
        return chrom, agg


class SummaryGraphBuilder(object):
    def __init__(self, solutions):
        self.solutions = solutions

    def plot(self, min_score=0.4, min_signal=0.2, colorizer=None, total_ion_chromatogram=None,
             base_peak_chromatogram=None):
        monosaccharides = set()

        for sol in self.solutions:
            if sol.glycan_composition:
                monosaccharides.update(map(str, sol.glycan_composition))

        label_abundant = plotting.AbundantLabeler(
            plotting.NGlycanLabelProducer(monosaccharides),
            max(sol.total_signal for sol in self.solutions if sol.score > min_score) * min_signal)

        if colorizer is None:
            colorizer = plotting.n_glycan_colorizer

        results = [sol for sol in self.solutions if sol.score > min_score and not sol.used_as_adduct]
        chrom = plotting.SmoothingChromatogramArtist(results, colorizer=colorizer).draw(label_function=label_abundant)

        if total_ion_chromatogram is not None:
            rt, intens = total_ion_chromatogram.as_arrays()
            chrom.draw_generic_chromatogram(
                "TIC", rt, intens, 'blue')
            chrom.ax.set_ylim(0, max(intens) * 1.1)

        if base_peak_chromatogram is not None:
            rt, intens = base_peak_chromatogram.as_arrays()
            chrom.draw_generic_chromatogram(
                "BPC", rt, intens, 'green')

        agg = plotting.AggregatedAbundanceArtist(results)
        agg.draw()
        return chrom, agg


ChromatogramProfiler.log_with_logger(logger)


class ScanDatabaseChromatogramProfiler(ChromatogramProfiler):
    def __init__(self, scan_db_path, composition_database_rules_path, storage_path=None,
                 sample_name=None, analysis_name=None, cache_handler_type=None):
        if cache_handler_type is None:
            cache_handler_type = NullScanCacheHandler

        self.scan_generator = DatabaseScanGenerator(scan_db_path, sample_name)

        self.storage_path = storage_path
        self.sample_name = sample_name
        self.analysis_name = analysis_name

        self.database_rules_path = composition_database_rules_path
        if isinstance(composition_database_rules_path, basestring):
            self.database = build_database(composition_database_rules_path)
        else:
            self.database = composition_database_rules_path

        self.cache_handler_type = cache_handler_type

        self.tracer = None
        self.adducts = []

        self.chromatograms = None
        self.solutions = None

        self._aggregate = True


GlycanProfiler = ChromatogramProfiler
ScanDatabaseGlycanProfiler = ScanDatabaseChromatogramProfiler
