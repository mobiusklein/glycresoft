import re
import time
import warnings
import logging

import glypy

from glycan_profiling import plotting
from glycan_profiling.database import build_database
from glycan_profiling.piped_deconvolve import ScanGenerator as PipedScanGenerator
from glycan_profiling.scoring import ChromatogramSolution, NetworkScoreDistributor, ChromatogramScorer
from glycan_profiling.trace import (
    IncludeUnmatchedTracer, ChromatogramFilter, join_mass_shifted,
    reverse_adduction_search, prune_bad_adduct_branches)
from brainpy import periodic_table
from ms_deisotope.averagine import Averagine, glycan as n_glycan_averagine


logger = logging.getLogger("glycan_profiler")


def validate_element(element):
    valid = element in periodic_table
    if not valid:
        warnings.warn("%r is not a valid element" % element)
    return valid


def parse_averagine_formula(formula):
    return Averagine({k: float(v) for k, v in re.findall(r"([A-Z][a-z]*)([0-9\.]*)", formula)
                      if float(v or 0) > 0 and validate_element(k)})


class GlycanProfiler(object):
    def __init__(self, mzml_path, database_rules_path, averagine=n_glycan_averagine, charge_range=(-1, -8),
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, ms1_deconvolution_args=None,
                 msn_deconvolution_args=None):
        if isinstance(averagine, basestring):
            averagine = parse_averagine_formula(averagine)

        self.mzml_path = mzml_path
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

    def search(self, mass_error_tolerance=1e-5, grouping_mass_error_tolerance=None, start_scan=None,
               max_scans=None, end_scan=None, adducts=None, truncate=False):

        start = time.time()

        if adducts is None:
            adducts = []
        self.adducts = adducts

        logger.info("Begin Chromatogram Tracing")
        self.scan_generator.configure_iteration(start_scan=start_scan, end_scan=end_scan, max_scans=max_scans)
        self.tracer = IncludeUnmatchedTracer(self.scan_generator, self.database, mass_error_tolerance)

        i = 0
        for case in self.tracer:
            logger.info("%d, %d, %s, %s, %d", i, case[1].index, case[1].scan_time, case[1].id, len(case[0]))
            i += 1
            if end_scan == case[1].id or i == max_scans:
                break

        self.tracer.commit()

        if grouping_mass_error_tolerance is None:
            grouping_mass_error_tolerance = mass_error_tolerance * 1.5

        self.build_chromatograms(mass_error_tolerance, grouping_mass_error_tolerance, adducts, truncate)
        logger.info("Tracing Complete (%r minutes elapsed)", (time.time() - start) / 60.)

    def build_chromatograms(self, mass_error_tolerance, grouping_mass_error_tolerance, adducts, truncate=False):
        logger.info("Post-Processing Chromatogram Traces (%0.3e, %0.3e, %r, %r)",
                    mass_error_tolerance, grouping_mass_error_tolerance, adducts, truncate)
        self.chromatograms = reverse_adduction_search(
            join_mass_shifted(
                ChromatogramFilter(self.tracer.chromatograms(
                    grouping_tolerance=grouping_mass_error_tolerance,
                    truncate=truncate)), adducts, mass_error_tolerance),
            adducts, mass_error_tolerance, self.database)

    def score(self, base_coef=0.8, support_coef=0.2, rt_delta=0.25, scoring_model=None):
        if scoring_model is None:
            scoring_model = ChromatogramScorer()
        self.solutions = []
        for case in ChromatogramFilter.process(self.chromatograms, delta_rt=rt_delta):
            try:
                if len(case) < 2:
                    continue
                self.solutions.append(ChromatogramSolution(case, scorer=scoring_model))
            except (IndexError, ValueError), e:
                print case, e, len(case)
                continue
        NetworkScoreDistributor(self.solutions, self.database.network).distribute(base_coef, support_coef)

        if self.adducts:
            hold = prune_bad_adduct_branches(self.solutions)
            self.solutions = []
            for case in hold:
                try:
                    if len(case) < 2:
                        continue
                    self.solutions.append(ChromatogramSolution(case, scorer=scoring_model))
                except (IndexError, ValueError, ZeroDivisionError), e:
                    print case, e, len(case)
                    continue
            NetworkScoreDistributor(self.solutions, self.database.network).distribute(base_coef, support_coef)

        self.solutions = ChromatogramFilter(sol for sol in self.solutions if sol.score > 1e-5)

    def plot(self, min_score=0.4, min_signal=0.2, colorizer=None, chromatogram_artist=None, include_tic=True):
        if chromatogram_artist is None:
            chromatogram_artist = plotting.SmoothingChromatogramArtist
        monosaccharides = set()

        for sol in self.solutions:
            if sol.composition:
                monosaccharides.update(map(str, glypy.GlycanComposition.parse(sol.composition)))

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
