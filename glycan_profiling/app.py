import glypy

from glycan_profiling import plotting
from glycan_profiling.database import build_database
from glycan_profiling.piped_deconvolve import ScanGenerator as PipedScanGenerator
from glycan_profiling.scoring import ChromatogramSolution, NetworkScoreDistributor
from glycan_profiling.trace import IncludeUnmatchedTracer, ChromatogramFilter, join_mass_shifted


class GlycanProfiler(object):
    def __init__(self, mzml_path, database_rules_path):
        self.mzml_path = mzml_path
        self.scan_generator = PipedScanGenerator(mzml_path)

        self.database_rules_path = database_rules_path
        if isinstance(database_rules_path, basestring):
            self.database = build_database(database_rules_path)
        else:
            self.database = database_rules_path

        self.tracer = None

        self.chromatograms = None
        self.solutions = None

    def search(self, mass_error_tolerance=1e-5, grouping_mass_error_tolerance=None, start_scan=None,
               max_scans=None, end_scan=None, adducts=None):

        if adducts is None:
            adducts = []

        self.scan_generator.configure_iteration(start_scan=start_scan, end_scan=end_scan, max_scans=max_scans)
        self.tracer = IncludeUnmatchedTracer(self.scan_generator, self.database, mass_error_tolerance)

        i = 0
        for case in self.tracer:
            print i, case[1].index, case[1].scan_time, case[1].id, len(case[0])
            i += 1
            if end_scan == case[1].id or i == max_scans:
                break

        if grouping_mass_error_tolerance is None:
            grouping_mass_error_tolerance = mass_error_tolerance * 1.5

        self.chromatograms = join_mass_shifted(
            ChromatogramFilter(self.tracer.chromatograms(
                grouping_tolerance=grouping_mass_error_tolerance)), adducts, mass_error_tolerance)

    def score(self, base_coef=0.8, support_coef=0.2, rt_delta=0.25):
        self.solutions = []
        for case in ChromatogramFilter.process(self.chromatograms, delta_rt=rt_delta):
            try:
                self.solutions.append(ChromatogramSolution(case))
                # print case
            except (IndexError, ValueError), e:
                print case, e, len(case)
                continue
        self.solutions = ChromatogramFilter(self.solutions)
        NetworkScoreDistributor(self.solutions, self.database.network).distribute(base_coef, support_coef)

    def plot(self, min_score=0.4, min_signal=0.2, colorizer=None, chromatogram_artist=None):
        if chromatogram_artist is None:
            chromatogram_artist = plotting.ChromatogramArtist
        monosaccharides = set()

        for sol in self.solutions:
            if sol.composition:
                monosaccharides.update(map(str, glypy.GlycanComposition.parse(sol.composition)))

        label_abundant = plotting.AbundantLabeler(
            plotting.NGlycanLabelProducer(monosaccharides),
            max(sol.total_signal for sol in self.solutions) * min_signal)

        if colorizer is None:
            colorizer = plotting.n_glycan_colorizer

        results = [sol for sol in self.solutions if sol.score > min_score]
        chrom = chromatogram_artist(results, colorizer=colorizer).draw(label_function=label_abundant)
        chrom.draw_generic_chromatogram(
            "TIC",
            map(self.tracer.scan_id_to_rt, self.tracer.total_ion_chromatogram),
            self.tracer.total_ion_chromatogram.values(), 'blue')
        chrom.ax.set_ylim(0, max(self.tracer.total_ion_chromatogram.values()) * 1.1)

        agg = plotting.AggregatedAbundanceArtist(results)
        agg.draw()
        return chrom, agg
