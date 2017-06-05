import re
import warnings

from glycan_profiling.database.disk_backed_database import (
    GlycanCompositionDiskBackedStructureDatabase,
    GlycopeptideDiskBackedStructureDatabase)

from glycan_profiling.serialize import (
    DatabaseScanDeserializer, AnalysisSerializer,
    AnalysisTypeEnum, GlycanCompositionChromatogramAnalysisSerializer,
    GlycopeptideMSMSAnalysisSerializer)

from glycan_profiling.piped_deconvolve import (
    ScanGenerator as PipedScanGenerator,
    RawScanGenerator as RawPipedScanGenerator)

from glycan_profiling.scoring import (
    ChromatogramSolution)

from glycan_profiling.trace import (
    ScanSink, ChromatogramExtractor,
    ChromatogramProcessor)

from glycan_profiling.models import GeneralScorer

from glycan_profiling.tandem import chromatogram_mapping
from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer
from glycan_profiling.tandem.glycopeptide.glycopeptide_matcher import GlycopeptideDatabaseSearchIdentifier
from glycan_profiling.tandem.glycopeptide import (
    identified_structure as identified_glycopeptide)


from glycan_profiling.scan_cache import (
    ThreadedDatabaseScanCacheHandler, ThreadedMzMLScanCacheHandler)

from glycan_profiling.task import TaskBase

from brainpy import periodic_table
from ms_deisotope.averagine import Averagine, glycan as n_glycan_averagine
from ms_deisotope.output.mzml import ProcessedMzMLDeserializer
from glycopeptidepy.utils.collectiontools import descending_combination_counter


def validate_element(element):
    valid = element in periodic_table
    if not valid:
        warnings.warn("%r is not a valid element" % element)
    return valid


def parse_averagine_formula(formula):
    return Averagine({k: float(v) for k, v in re.findall(r"([A-Z][a-z]*)([0-9\.]*)", formula)
                      if float(v or 0) > 0 and validate_element(k)})


class SampleConsumer(TaskBase):
    MS1_ISOTOPIC_PATTERN_WIDTH = 0.95
    MS1_IGNORE_BELOW = 0.05
    MSN_ISOTOPIC_PATTERN_WIDTH = 0.80
    MSN_IGNORE_BELOW = 0.05

    def __init__(self, ms_file, averagine=n_glycan_averagine, charge_range=(-1, -8),
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, ms1_deconvolution_args=None,
                 msn_deconvolution_args=None, start_scan_id=None, end_scan_id=None, storage_path=None,
                 sample_name=None, cache_handler_type=None, n_processes=5,
                 extract_only_tandem_envelopes=False, ignore_tandem_scans=False):

        if cache_handler_type is None:
            cache_handler_type = ThreadedDatabaseScanCacheHandler
        if isinstance(averagine, basestring):
            averagine = parse_averagine_formula(averagine)

        self.ms_file = ms_file
        self.storage_path = storage_path
        self.sample_name = sample_name

        self.n_processes = n_processes
        self.cache_handler_type = cache_handler_type
        self.extract_only_tandem_envelopes = extract_only_tandem_envelopes

        n_helpers = max(self.n_processes - 1, 0)
        self.scan_generator = PipedScanGenerator(
            ms_file, averagine=averagine, charge_range=charge_range,
            number_of_helpers=n_helpers,
            ms1_peak_picking_args=ms1_peak_picking_args,
            msn_peak_picking_args=msn_peak_picking_args,
            ms1_deconvolution_args=ms1_deconvolution_args,
            msn_deconvolution_args=msn_deconvolution_args,
            extract_only_tandem_envelopes=extract_only_tandem_envelopes,
            ignore_tandem_scans=ignore_tandem_scans)

        self.start_scan_id = start_scan_id
        self.end_scan_id = end_scan_id

        self.sample_run = None

    def run(self):
        self.log("Initializing Generator")
        self.scan_generator.configure_iteration(self.start_scan_id, self.end_scan_id)
        self.log("Setting Sink")
        sink = ScanSink(self.scan_generator, self.cache_handler_type)
        self.log("Initializing Cache")
        sink.configure_cache(self.storage_path, self.sample_name, self.scan_generator)

        self.log("Begin Processing")
        last_scan_time = 0
        last_scan_index = 0
        for scan in sink:
            if scan.scan_time - last_scan_time > 1.0:
                self.log("Processed %s (time: %f)" % (
                    scan.id, scan.scan_time,))
                if last_scan_index != 0:
                    self.log("Count Since Last Log: %d" % (scan.index - last_scan_index,))
                last_scan_time = scan.scan_time
                last_scan_index = scan.index
        self.log("Finished Recieving Scans")
        sink.complete()
        self.log("Completed Sample %s" % (self.sample_name,))
        sink.commit()

        self.sample_run = sink.sample_run


class CentroidingSampleConsumer(SampleConsumer):
    def __init__(self, ms_file, averagine=n_glycan_averagine, charge_range=(-1, -8),
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, start_scan_id=None,
                 end_scan_id=None, storage_path=None, sample_name=None, cache_handler_type=None,
                 n_processes=5, extract_only_tandem_envelopes=False):

        if cache_handler_type is None:
            cache_handler_type = ThreadedDatabaseScanCacheHandler
        if isinstance(averagine, basestring):
            averagine = parse_averagine_formula(averagine)

        self.ms_file = ms_file
        self.storage_path = storage_path
        self.sample_name = sample_name

        self.n_processes = n_processes
        self.cache_handler_type = cache_handler_type
        self.extract_only_tandem_envelopes = extract_only_tandem_envelopes

        n_helpers = max(self.n_processes - 1, 0)
        self.scan_generator = RawPipedScanGenerator(
            ms_file,
            number_of_helpers=n_helpers,
            ms1_peak_picking_args=ms1_peak_picking_args,
            msn_peak_picking_args=msn_peak_picking_args,
            extract_only_tandem_envelopes=extract_only_tandem_envelopes)

        self.start_scan_id = start_scan_id
        self.end_scan_id = end_scan_id

        self.sample_run = None


class GlycanChromatogramAnalyzer(TaskBase):
    @staticmethod
    def expand_adducts(adduct_counts):
        counts = descending_combination_counter(adduct_counts)
        combinations = []
        for combo in counts:
            scaled = []
            for k, v in combo.items():
                if v == 0:
                    continue
                scaled.append(k * v)
            if scaled:
                base = scaled[0]
                for ad in scaled[1:]:
                    base += ad
                combinations.append(base)
        return combinations

    def __init__(self, database_connection, hypothesis_id, sample_run_id, adducts=None,
                 mass_error_tolerance=1e-5, grouping_error_tolerance=1.5e-5,
                 scoring_model=GeneralScorer, network_sharing=0.2, minimum_mass=500.,
                 analysis_name=None):
        self.database_connection = database_connection
        self.hypothesis_id = hypothesis_id
        self.sample_run_id = sample_run_id
        self.mass_error_tolerance = mass_error_tolerance
        self.grouping_error_tolerance = grouping_error_tolerance
        self.minimum_mass = minimum_mass
        self.scoring_model = scoring_model
        self.network_sharing = network_sharing
        self.adducts = adducts
        self.analysis_name = analysis_name
        self.analysis = None

    def save_solutions(self, solutions, extractor, database):
        if self.analysis_name is None:
            return
        self.log('Saving solutions')
        analysis_saver = AnalysisSerializer(
            self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(extractor.peak_mapping)
        analysis_saver.set_analysis_type(AnalysisTypeEnum.glycan_lc_ms.name)

        analysis_saver.set_parameters({
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "mass_error_tolerance": self.mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "network_sharing": self.network_sharing,
            "adducts": [adduct.name for adduct in self.adducts],
            "minimum_mass": self.minimum_mass,
        })

        n = len(solutions)
        i = 0
        for chroma in solutions:
            i += 1
            if i % 100 == 0:
                self.log("%0.2f%% of Chromatograms Saved (%d/%d)" % (i * 100. / n, i, n))
            if chroma.composition:
                analysis_saver.save_glycan_composition_chromatogram_solution(chroma)
            else:
                analysis_saver.save_unidentified_chromatogram_solution(chroma)

        self.analysis = analysis_saver.analysis
        analysis_saver.commit()

    def make_peak_loader(self):
        peak_loader = DatabaseScanDeserializer(
            self.database_connection, sample_run_id=self.sample_run_id)
        return peak_loader

    def make_database(self):
        database = GlycanCompositionDiskBackedStructureDatabase(
            self.database_connection, self.hypothesis_id)
        return database

    def make_chromatogram_extractor(self, peak_loader):
        extractor = ChromatogramExtractor(
            peak_loader, grouping_tolerance=self.grouping_error_tolerance,
            minimum_mass=self.minimum_mass)
        return extractor

    def make_chromatogram_processor(self, extractor, database):
        proc = ChromatogramProcessor(
            extractor, database, mass_error_tolerance=self.mass_error_tolerance,
            adducts=self.adducts, scoring_model=self.scoring_model,
            network_sharing=self.network_sharing)
        return proc

    def run(self):
        peak_loader = self.make_peak_loader()
        database = self.make_database()
        extractor = self.make_chromatogram_extractor(peak_loader)
        proc = self.make_chromatogram_processor(extractor, database)
        proc.run()
        self.save_solutions(proc.solutions, extractor, database)
        return proc


class MzMLGlycanChromatogramAnalyzer(GlycanChromatogramAnalyzer):
    def __init__(self, database_connection, hypothesis_id, sample_path, output_path,
                 adducts=None, mass_error_tolerance=1e-5, grouping_error_tolerance=1.5e-5,
                 scoring_model=None, network_sharing=0.2, minimum_mass=500.,
                 analysis_name=None):
        super(MzMLGlycanChromatogramAnalyzer, self).__init__(
            database_connection, hypothesis_id, -1, adducts,
            mass_error_tolerance, grouping_error_tolerance,
            scoring_model, network_sharing, minimum_mass,
            analysis_name)
        self.sample_path = sample_path
        self.output_path = output_path

    def make_peak_loader(self):
        peak_loader = ProcessedMzMLDeserializer(self.sample_path)
        return peak_loader

    def save_solutions(self, solutions, extractor, database):
        if self.analysis_name is None:
            return
        self.log('Saving solutions')

        exporter = GlycanCompositionChromatogramAnalysisSerializer(
            self.output_path, self.analysis_name, extractor.peak_loader.sample_run,
            solutions, database, extractor)

        exporter.run()
        exporter.set_parameters({
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_path,
            "sample_path": self.sample_path,
            "sample_name": extractor.peak_loader.sample_run.name,
            "mass_error_tolerance": self.mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "network_sharing": self.network_sharing,
            "adducts": [adduct.name for adduct in self.adducts],
            "minimum_mass": self.minimum_mass,
        })
        self.analysis = exporter.analysis


class GlycopeptideLCMSMSAnalyzer(TaskBase):
    def __init__(self, database_connection, hypothesis_id, sample_run_id,
                 analysis_name=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                 tandem_scoring_model=None, minimum_mass=1000., save_unidentified=False,
                 oxonium_threshold=0.05, scan_transformer=None, n_processes=5):
        if tandem_scoring_model is None:
            tandem_scoring_model = CoverageWeightedBinomialScorer
        if peak_shape_scoring_model is None:
            peak_shape_scoring_model = GeneralScorer
        if scan_transformer is None:
            def scan_transformer(x):
                return x

        self.database_connection = database_connection
        self.hypothesis_id = hypothesis_id
        self.sample_run_id = sample_run_id
        self.analysis_name = analysis_name
        self.mass_error_tolerance = mass_error_tolerance
        self.msn_mass_error_tolerance = msn_mass_error_tolerance
        self.grouping_error_tolerance = grouping_error_tolerance
        self.psm_fdr_threshold = psm_fdr_threshold
        self.peak_shape_scoring_model = peak_shape_scoring_model
        self.tandem_scoring_model = tandem_scoring_model
        self.analysis = None
        self.analysis_id = None
        self.minimum_mass = minimum_mass
        self.save_unidentified = save_unidentified
        self.minimum_oxonium_ratio = oxonium_threshold
        self.scan_transformer = scan_transformer
        self.n_processes = n_processes

    def make_peak_loader(self):
        peak_loader = DatabaseScanDeserializer(
            self.database_connection, sample_run_id=self.sample_run_id)
        return peak_loader

    def make_database(self):
        database = GlycopeptideDiskBackedStructureDatabase(
            self.database_connection, self.hypothesis_id)
        return database

    def make_chromatogram_extractor(self, peak_loader):
        extractor = ChromatogramExtractor(
            peak_loader, grouping_tolerance=self.grouping_error_tolerance,
            minimum_mass=self.minimum_mass)
        return extractor

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [o.product for o in prec_info]
        return msms_scans

    def make_search_engine(self, msms_scans, database, peak_loader):
        searcher = GlycopeptideDatabaseSearchIdentifier(
            msms_scans, self.tandem_scoring_model, database,
            peak_loader.convert_scan_id_to_retention_time,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            scan_transformer=self.scan_transformer,
            n_processes=self.n_processes)
        return searcher

    def do_search(self, searcher):
        target_hits, decoy_hits = searcher.search(
            precursor_error_tolerance=self.mass_error_tolerance,
            error_tolerance=self.msn_mass_error_tolerance)
        return target_hits, decoy_hits

    def estimate_fdr(self, searcher, target_hits, decoy_hits):
        searcher.target_decoy(target_hits, decoy_hits)

    def map_chromatograms(self, searcher, extractor, target_hits):
        chroma_with_sols, orphans = searcher.map_to_chromatograms(
            tuple(extractor), target_hits, self.mass_error_tolerance,
            threshold_fn=lambda x: x.q_value < self.psm_fdr_threshold)
        merged = chromatogram_mapping.aggregate_by_assigned_entity(chroma_with_sols)
        return merged, orphans

    def score_chromatograms(self, merged):
        chroma_scoring_model = self.peak_shape_scoring_model
        scored_merged = []
        n = len(merged)
        i = 0
        for c in merged:
            i += 1
            if i % 500 == 0:
                self.log("%0.2f%% chromatograms evaluated (%d/%d) %r" % (i * 100. / n, i, n, c))
            try:
                scored_merged.append(ChromatogramSolution(c, scorer=chroma_scoring_model))
            except (IndexError, ValueError) as e:
                self.log("Could not score chromatogram %r due to %s" % (c, e))
                scored_merged.append(ChromatogramSolution(c, score=0.0))
        return scored_merged

    def assign_consensus(self, scored_merged, orphans):
        self.log("Assigning consensus glycopeptides to spectrum clusters")
        assigned_list = list(scored_merged)
        assigned_list.extend(orphans)
        gps, unassigned = identified_glycopeptide.extract_identified_structures(
            assigned_list, lambda x: x.q_value < self.psm_fdr_threshold)
        return gps, unassigned

    def run(self):
        peak_loader = self.make_peak_loader()
        database = self.make_database()
        extractor = self.make_chromatogram_extractor(peak_loader)

        self.log("Loading MS/MS")

        msms_scans = self.load_msms(peak_loader)

        # Traditional LC-MS/MS Database Search
        searcher = self.make_search_engine(msms_scans, database, peak_loader)
        target_hits, decoy_hits = self.do_search(searcher)

        if len(target_hits) == 0:
            self.log("No target matches were found.")
            return [], [], [], []

        self.estimate_fdr(searcher, target_hits, decoy_hits)
        n_below = 0
        for target in target_hits:
            if target.q_value <= self.psm_fdr_threshold:
                n_below += 1
        self.log("%d spectrum matches accepted" % (n_below,))

        # Map MS/MS solutions to chromatograms.
        self.log("Building and Mapping Chromatograms")
        merged, orphans = self.map_chromatograms(searcher, extractor, target_hits)

        if not self.save_unidentified:
            merged = [chroma for chroma in merged if chroma.composition is not None]

        # Score chromatograms, both matched and unmatched
        self.log("Scoring chromatograms")
        scored_merged = self.score_chromatograms(merged)

        gps, unassigned = self.assign_consensus(scored_merged, orphans)

        self.log("Saving solutions (%d identified glycopeptides)" % (len(gps),))
        self.save_solutions(gps, unassigned, extractor, database)
        return gps, unassigned, target_hits, decoy_hits

    def save_solutions(self, identified_glycopeptides, unassigned_chromatograms,
                       chromatogram_extractor, database):
        if self.analysis_name is None:
            return
        analysis_saver = AnalysisSerializer(self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(chromatogram_extractor.peak_mapping)
        analysis_saver.set_analysis_type(AnalysisTypeEnum.glycopeptide_lc_msms.name)
        analysis_saver.set_parameters({
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "mass_error_tolerance": self.mass_error_tolerance,
            "fragment_error_tolerance": self.msn_mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "psm_fdr_threshold": self.psm_fdr_threshold,
            "minimum_mass": self.minimum_mass,
        })

        analysis_saver.save_glycopeptide_identification_set(identified_glycopeptides)
        if self.save_unidentified:
            i = 0
            last = 0
            interval = 100
            n = len(unassigned_chromatograms)
            for chroma in unassigned_chromatograms:
                i += 1
                if (i - last > interval):
                    self.log("Saving Unidentified Chromatogram %d/%d (%0.2f%%)" % (i, n, (i * 100. / n)))
                    last = i
                analysis_saver.save_unidentified_chromatogram_solution(chroma)

        analysis_saver.commit()
        self.analysis = analysis_saver.analysis
        self.analysis_id = analysis_saver.analysis_id


class ScanStub(object):
    """A stub for holding precursor information and
    giving a Scan-like interface for accessing just that
    information. Provides a serialized-like interface
    which clients can use to load the real scan.

    Attributes
    ----------
    id : str
        The scan ID for the proxied scan
    precursor_information : PrecursorInformation
        The information describing the relevant
        metadata for scheduling when and where this
        scan should be processed, where actual loading
        will occur.
    bind : MzMLLoader
        A resource to use to load scans with by scan id.
    """
    def __init__(self, precursor_information, bind):
        self.id = precursor_information.product_scan_id
        self.precursor_information = precursor_information
        self.bind = bind

    def convert(self, *args, **kwargs):
        try:
            return self.bind.get_scan_by_id(self.id)
        except AttributeError:
            raise KeyError(self.id)


class MzMLGlycopeptideLCMSMSAnalyzer(GlycopeptideLCMSMSAnalyzer):
    def __init__(self, database_connection, hypothesis_id, sample_path, output_path,
                 analysis_name=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                 tandem_scoring_model=None, minimum_mass=1000., save_unidentified=False,
                 oxonium_threshold=0.05, scan_transformer=None, n_processes=5):
        super(MzMLGlycopeptideLCMSMSAnalyzer, self).__init__(
            database_connection,
            hypothesis_id, -1,
            analysis_name, grouping_error_tolerance,
            mass_error_tolerance, msn_mass_error_tolerance,
            psm_fdr_threshold, peak_shape_scoring_model,
            tandem_scoring_model, minimum_mass,
            save_unidentified, oxonium_threshold,
            scan_transformer,
            n_processes)
        self.sample_path = sample_path
        self.output_path = output_path

    def make_peak_loader(self):
        peak_loader = ProcessedMzMLDeserializer(self.sample_path)
        return peak_loader

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [ScanStub(o, peak_loader) for o in prec_info]
        return msms_scans

    def save_solutions(self, identified_glycopeptides, unassigned_chromatograms,
                       chromatogram_extractor, database):
        if self.analysis_name is None:
            return
        exporter = GlycopeptideMSMSAnalysisSerializer(
            self.output_path,
            self.analysis_name,
            chromatogram_extractor.peak_loader.sample_run,
            identified_glycopeptides,
            unassigned_chromatograms,
            database,
            chromatogram_extractor)

        exporter.run()

        exporter.set_parameters({
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "sample_path": self.sample_path,
            "sample_name": chromatogram_extractor.peak_loader.sample_run.name,
            "mass_error_tolerance": self.mass_error_tolerance,
            "fragment_error_tolerance": self.msn_mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "psm_fdr_threshold": self.psm_fdr_threshold,
            "minimum_mass": self.minimum_mass,
        })

        self.analysis = exporter.analysis
