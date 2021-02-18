'''High level analytical pipeline implementations.

Each class is designed to encapsulate a single broad task, i.e.
LC-MS/MS deconvolution or structure identification
'''
import os
from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

import ms_peak_picker

import ms_deisotope
from ms_deisotope.output.mzml import ProcessedMzMLDeserializer

import glypy
from glypy.utils import Enum

from glycopeptidepy.utils.collectiontools import descending_combination_counter


from glycan_profiling.database.disk_backed_database import (
    GlycanCompositionDiskBackedStructureDatabase,
    GlycopeptideDiskBackedStructureDatabase)

from glycan_profiling.database.analysis import (
    GlycanCompositionChromatogramAnalysisSerializer,
    DynamicGlycopeptideMSMSAnalysisSerializer,
    GlycopeptideMSMSAnalysisSerializer)

from glycan_profiling.serialize import (
    DatabaseScanDeserializer, AnalysisSerializer,
    AnalysisTypeEnum, object_session)

from glycan_profiling.piped_deconvolve import (
    ScanGenerator as PipedScanGenerator)

from glycan_profiling.scoring import (
    ChromatogramSolution)

from glycan_profiling.trace import (
    ScanSink,
    ChromatogramExtractor,
    LogitSumChromatogramProcessor,
    LaplacianRegularizedChromatogramProcessor,
    ChromatogramEvaluator)

from glycan_profiling import config # pylint: disable=unused-import

from glycan_profiling.chromatogram_tree import ChromatogramFilter, SimpleChromatogram, Unmodified

from glycan_profiling.models import GeneralScorer, get_feature

from glycan_profiling.structure import ScanStub
from glycan_profiling.structure.structure_loader import (
    oxonium_ion_cache, CachingStubGlycopeptideStrategy)

from glycan_profiling.tandem import chromatogram_mapping
from glycan_profiling.tandem.target_decoy import TargetDecoySet
from glycan_profiling.tandem.temp_store import TempFileManager

from glycan_profiling.tandem.spectrum_match.solution_set import QValueRetentionStrategy

from glycan_profiling.tandem.glycopeptide.scoring import (
    LogIntensityScorer,
    CoverageWeightedBinomialScorer,
    CoverageWeightedBinomialModelTree)

from glycan_profiling.tandem.glycopeptide.glycopeptide_matcher import (
    GlycopeptideDatabaseSearchIdentifier,
    ExclusiveGlycopeptideDatabaseSearchComparer)

from glycan_profiling.tandem.glycopeptide.dynamic_generation import (
    MultipartGlycopeptideIdentifier, GlycopeptideFDREstimationStrategy)

from glycan_profiling.tandem.glycopeptide import (
    identified_structure as identified_glycopeptide)

from glycan_profiling.tandem.glycan.composition_matching import SignatureIonMapper
from glycan_profiling.tandem.glycan.scoring.signature_ion_scoring import SignatureIonScorer


from glycan_profiling.scan_cache import (
    ThreadedMzMLScanCacheHandler)

from glycan_profiling.task import TaskBase
from glycan_profiling import serialize


debug_mode = bool(os.environ.get('GLYCRESOFTDEBUG', False))


class SampleConsumer(TaskBase):
    """Implements the LC-MS/MS sample deconvolution pipeline, taking an arbitrary
    MS data file providing MS1 and MSn scans and produces a new mzML file with the
    deisotoped and charge state deconvolved data from each spectrum in it.

    Makes heavy use of :mod:`ms_deisotope` and :mod:`ms_peak_picker`
    """

    MS1_ISOTOPIC_PATTERN_WIDTH = 0.95
    MS1_IGNORE_BELOW = 0.05
    MSN_ISOTOPIC_PATTERN_WIDTH = 0.80
    MSN_IGNORE_BELOW = 0.05

    MS1_SCORE_THRESHOLD = 20.0
    MSN_SCORE_THRESHOLD = 10.0

    def __init__(self, ms_file,
                 ms1_peak_picking_args=None, msn_peak_picking_args=None, ms1_deconvolution_args=None,
                 msn_deconvolution_args=None, start_scan_id=None, end_scan_id=None, storage_path=None,
                 sample_name=None, cache_handler_type=None, n_processes=5,
                 extract_only_tandem_envelopes=False, ignore_tandem_scans=False,
                 ms1_averaging=0, deconvolute=True):

        if cache_handler_type is None:
            cache_handler_type = ThreadedMzMLScanCacheHandler

        self.ms_file = ms_file
        self.storage_path = storage_path
        self.sample_name = sample_name

        self.n_processes = n_processes
        self.cache_handler_type = cache_handler_type
        self.extract_only_tandem_envelopes = extract_only_tandem_envelopes
        self.ignore_tandem_scans = ignore_tandem_scans
        self.ms1_averaging = ms1_averaging
        self.ms1_processing_args = {
            "peak_picking": ms1_peak_picking_args,
        }
        self.msn_processing_args = {
            "peak_picking": msn_peak_picking_args,
        }

        self.deconvolute = deconvolute

        if deconvolute:
            self.ms1_processing_args["deconvolution"] = ms1_deconvolution_args
            self.msn_processing_args["deconvolution"] = msn_deconvolution_args

        n_helpers = max(self.n_processes - 1, 0)
        self.scan_generator = PipedScanGenerator(
            ms_file,
            number_of_helpers=n_helpers,
            ms1_peak_picking_args=ms1_peak_picking_args,
            msn_peak_picking_args=msn_peak_picking_args,
            ms1_deconvolution_args=ms1_deconvolution_args,
            msn_deconvolution_args=msn_deconvolution_args,
            extract_only_tandem_envelopes=extract_only_tandem_envelopes,
            ignore_tandem_scans=ignore_tandem_scans,
            ms1_averaging=ms1_averaging, deconvolute=deconvolute)

        self.start_scan_id = start_scan_id
        self.end_scan_id = end_scan_id

        self.sample_run = None

    @staticmethod
    def default_processing_configuration(averagine=ms_deisotope.glycopeptide, msn_averagine=None):
        """Create the default scan-level processing parameters for the pipeline if
        not provided.

        This function is mainly useful for testing and debugging as this information is usually
        provided by the user.

        Parameters
        ----------
        averagine : :class:`ms_deisotope.Averagine`, optional
            The averagine model used for MS1 spectra (the default is ms_deisotope.glycopeptide)
        msn_averagine : :class:`ms_deisotope.Averagine`, optional
            The averagine model used for MSn spectra (the default is None, which will default to the
            same as the MS1 model)

        Returns
        -------
        :class:`tuple` of 4 :class:`dict`
        """

        if msn_averagine is None:
            msn_averagine = averagine

        ms1_peak_picking_args = {
            "transforms": [
                ms_peak_picker.scan_filter.FTICRBaselineRemoval(
                    scale=5.0, window_length=2),
                ms_peak_picker.scan_filter.SavitskyGolayFilter()
            ]
        }

        ms1_deconvolution_args = {
            "scorer": ms_deisotope.scoring.PenalizedMSDeconVFitter(20, 2.),
            "max_missed_peaks": 3,
            "averagine": averagine,
            "truncate_after": SampleConsumer.MS1_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MS1_IGNORE_BELOW,
            "deconvoluter_type": ms_deisotope.AveraginePeakDependenceGraphDeconvoluter
        }

        msn_peak_picking_args = {}

        msn_deconvolution_args = {
            "scorer": ms_deisotope.scoring.MSDeconVFitter(10),
            "averagine": msn_averagine,
            "max_missed_peaks": 1,
            "truncate_after": SampleConsumer.MSN_ISOTOPIC_PATTERN_WIDTH,
            "ignore_below": SampleConsumer.MSN_IGNORE_BELOW
        }

        return (ms1_peak_picking_args, msn_peak_picking_args,
                ms1_deconvolution_args, msn_deconvolution_args)

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
        i = 0
        for scan in sink:
            i += 1
            if (scan.scan_time - last_scan_time > 1.0) or (i % 1000 == 0):
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


class ChromatogramSummarizer(TaskBase):
    """Implement the simple diagnostic chromatogram extraction pipeline which
    given a deconvoluted mzML file produced by :class:`SampleConsumer` will build
    aggregated extracted ion chromatograms for each distinct mass over time.

    Unlike most of the pipelines here, this task does not currently save its own output,
    instead returning it to the caller of it's :meth:`run` method.
    """
    def __init__(self, mzml_path, threshold_percentile=90, minimum_mass=300.0, extract_signatures=True, evaluate=False,
                 chromatogram_scoring_model=None):
        if chromatogram_scoring_model is None:
            chromatogram_scoring_model = GeneralScorer
        self.mzml_path = mzml_path
        self.threshold_percentile = threshold_percentile
        self.minimum_mass = minimum_mass
        self.extract_signatures = extract_signatures
        self.intensity_threshold = 0.0
        self.chromatogram_scoring_model = chromatogram_scoring_model
        self.should_evaluate = evaluate

    def make_scan_loader(self):
        '''Create a reader for the deconvoluted LC-MS data file
        '''
        scan_loader = ProcessedMzMLDeserializer(self.mzml_path)
        return scan_loader

    def estimate_intensity_threshold(self, scan_loader):
        '''Given a reader with an extended index, build the MS1 peak
        intensity distribution to estimate the global intensity threshold
        to use when extracting chromatograms.
        '''
        acc = []
        for scan_id in scan_loader.extended_index.ms1_ids:
            header = scan_loader.get_scan_header_by_id(scan_id)
            acc.extend(header.arrays.intensity)
        self.intensity_threshold = np.percentile(acc, self.threshold_percentile)
        return self.intensity_threshold

    def extract_chromatograms(self, scan_loader):
        '''Perform the chromatogram extraction process.
        '''
        extractor = ChromatogramExtractor(
            scan_loader, minimum_intensity=self.intensity_threshold,
            minimum_mass=self.minimum_mass)
        chroma = extractor.run()
        return chroma, extractor.total_ion_chromatogram, extractor.base_peak_chromatogram

    def extract_signature_ion_traces(self, scan_loader):
        '''Skim the MSn spectra to look for oxonium ion signatures over
        time.
        '''
        from glycan_profiling.tandem.oxonium_ions import standard_oxonium_ions
        window_width = 0.01
        ox_time = []
        ox_current = []
        for scan_id in scan_loader.extended_index.msn_ids:
            try:
                scan = scan_loader.get_scan_header_by_id(scan_id)
            except AttributeError:
                self.log("Unable to resolve scan id %r" % scan_id)
            total = 0
            for ion in standard_oxonium_ions:
                mid = ion.mass() + 1.007
                lo = mid - window_width
                hi = mid + window_width
                sig_slice = scan.arrays.between_mz(lo, hi)
                total += sig_slice.intensity.sum()
            ox_time.append(scan.scan_time)
            ox_current.append(total)
        oxonium_ion_chromatogram = SimpleChromatogram(zip(ox_time, ox_current))
        return oxonium_ion_chromatogram

    def evaluate_chromatograms(self, chromatograms):
        evaluator = ChromatogramEvaluator(self.chromatogram_scoring_model)
        solutions = evaluator.score(chromatograms)
        return solutions

    def run(self):
        scan_loader = self.make_scan_loader()
        self.log("... Estimating Intensity Threshold")
        self.estimate_intensity_threshold(scan_loader)
        chroma, total_ion_chromatogram, base_peak_chromatogram = self.extract_chromatograms(
            scan_loader)
        oxonium_ion_chromatogram = None
        if self.extract_signatures:
            oxonium_ion_chromatogram = self.extract_signature_ion_traces(scan_loader)
        if self.should_evaluate:
            chroma = self.evaluate_chromatograms(chroma)
        return chroma, (total_ion_chromatogram, base_peak_chromatogram, oxonium_ion_chromatogram)


class GlycanChromatogramAnalyzer(TaskBase):
    """Analyze glycan LC-MS profiling data, assigning glycan compositions
    to extracted chromatograms.

    The base implementation targets the legacy deconvoluted spectrum
    database format. See :class:`MzMLGlycanChromatogramAnalyzer` for the
    newer implementation targeting the mzML file produced by :class:`SampleConsumer`.
    """

    @staticmethod
    def expand_mass_shifts(mass_shift_counts, crossproduct=True, limit=None):
        if limit is None:
            limit = float('inf')
        combinations = []
        if crossproduct:
            counts = descending_combination_counter(mass_shift_counts)
            for combo in counts:
                scaled = []
                if sum(combo.values()) > limit:
                    continue
                for k, v in combo.items():
                    if v == 0:
                        continue
                    scaled.append(k * v)
                if scaled:
                    base = scaled[0]
                    for ad in scaled[1:]:
                        base += ad
                    combinations.append(base)
        else:
            for k, total in mass_shift_counts.items():
                for t in range(1, total + 1):
                    combinations.append(k * t)
        return combinations

    def __init__(self, database_connection, hypothesis_id, sample_run_id, mass_shifts=None,
                 mass_error_tolerance=1e-5, grouping_error_tolerance=1.5e-5,
                 scoring_model=GeneralScorer, minimum_mass=500., regularize=None,
                 regularization_model=None, network=None, analysis_name=None,
                 delta_rt=0.5, require_msms_signature=0, msn_mass_error_tolerance=2e-5,
                 n_processes=4):

        if mass_shifts is None:
            mass_shifts = []

        self.database_connection = database_connection
        self.hypothesis_id = hypothesis_id
        self.sample_run_id = sample_run_id

        self.mass_error_tolerance = mass_error_tolerance
        self.grouping_error_tolerance = grouping_error_tolerance
        self.msn_mass_error_tolerance = msn_mass_error_tolerance

        self.scoring_model = scoring_model
        self.regularize = regularize

        self.network = network
        self.regularization_model = regularization_model

        self.minimum_mass = minimum_mass
        self.delta_rt = delta_rt
        self.mass_shifts = mass_shifts

        self.require_msms_signature = require_msms_signature

        self.analysis_name = analysis_name
        self.analysis = None
        self.n_processes = n_processes

    def save_solutions(self, solutions, extractor, database, evaluator):
        if self.analysis_name is None:
            return
        self.log('Saving solutions')
        analysis_saver = AnalysisSerializer(
            self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(extractor.peak_mapping)
        analysis_saver.set_analysis_type(AnalysisTypeEnum.glycan_lc_ms.name)

        param_dict = {
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "mass_error_tolerance": self.mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "mass_shifts": [mass_shift.name for mass_shift in self.mass_shifts],
            "minimum_mass": self.minimum_mass,
            "require_msms_signature": self.require_msms_signature,
            "msn_mass_error_tolerance": self.msn_mass_error_tolerance
        }

        evaluator.update_parameters(param_dict)

        analysis_saver.set_parameters(param_dict)

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

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [
            o.product for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def make_database(self):
        combn_size = len(self.mass_shifts)
        database = GlycanCompositionDiskBackedStructureDatabase(
            self.database_connection, self.hypothesis_id, cache_size=combn_size)
        return database

    def make_chromatogram_extractor(self, peak_loader):
        extractor = ChromatogramExtractor(
            peak_loader, grouping_tolerance=self.grouping_error_tolerance,
            minimum_mass=self.minimum_mass, delta_rt=self.delta_rt)
        return extractor

    def make_chromatogram_processor(self, extractor, database):
        if self.regularize is not None or self.regularization_model is not None:
            proc = LaplacianRegularizedChromatogramProcessor(
                extractor, database, network=self.network,
                mass_error_tolerance=self.mass_error_tolerance,
                mass_shifts=self.mass_shifts, scoring_model=self.scoring_model,
                delta_rt=self.delta_rt, smoothing_factor=self.regularize,
                regularization_model=self.regularization_model,
                peak_loader=extractor.peak_loader)
        else:
            proc = LogitSumChromatogramProcessor(
                extractor, database, mass_error_tolerance=self.mass_error_tolerance,
                mass_shifts=self.mass_shifts, scoring_model=self.scoring_model,
                delta_rt=self.delta_rt,
                peak_loader=extractor.peak_loader)
        return proc

    def make_mapper(self, chromatograms, peak_loader, msms_scans=None, default_glycan_composition=None,
                    scorer_type=SignatureIonScorer):
        mapper = SignatureIonMapper(
            msms_scans, chromatograms, peak_loader.convert_scan_id_to_retention_time,
            self.mass_shifts, self.minimum_mass, batch_size=1000,
            default_glycan_composition=default_glycan_composition,
            scorer_type=scorer_type, n_processes=self.n_processes)
        return mapper

    def annotate_matches_with_msms(self, chromatograms, peak_loader, msms_scans, database):
        """Map MSn scans to chromatograms matched by precursor mass, and
        evaluate each glycan compostion-spectrum match

        Parameters
        ----------
        chromatograms : ChromatogramFilter
            Description
        peak_loader : RandomAccessScanIterator
            Description
        msms_scans : list
            Description
        database : SearchableMassCollection
            Description

        Returns
        -------
        ChromatogramFilter
            The chromatograms with matched and scored MSn scans attached to them
        """
        default_glycan_composition = glypy.GlycanComposition(
            database.hypothesis.monosaccharide_bounds())
        mapper = self.make_mapper(
            chromatograms, peak_loader, msms_scans, default_glycan_composition)
        self.log("Mapping MS/MS")
        mapped_matches = mapper.map_to_chromatograms(self.mass_error_tolerance)
        self.log("Evaluating MS/MS")
        annotate_matches = mapper.score_mapped_tandem(
            mapped_matches, error_tolerance=self.msn_mass_error_tolerance, include_compound=True)
        return annotate_matches

    def process_chromatograms(self, processor, peak_loader, database):
        """Extract, match and evaluate chromatograms against the glycan database.

        If MSn are available and required, then MSn scan will be extracted
        and mapped onto chromatograms, and search each MSn scan with the
        pseudo-fragments of the glycans matching the chromatograms they
        map to.

        Parameters
        ----------
        processor : ChromatgramProcessor
            The container responsible for carrying out the matching
            and evaluating of chromatograms
        peak_loader : RandomAccessScanIterator
            An object which can be used iterate over MS scans
        database : SearchableMassCollection
            The database of glycan compositions to serch against
        """
        if self.require_msms_signature > 0:
            self.log("Extracting MS/MS")
            msms_scans = self.load_msms(peak_loader)
            if len(msms_scans) == 0:
                self.log("No MS/MS scans present. Ignoring requirement.")
                processor.run()
            else:
                matches = processor.match_compositions()
                annotated_matches = self.annotate_matches_with_msms(
                    matches, peak_loader, msms_scans, database)
                # filter out those matches which do not have sufficient signature ion signal
                # from MS2 to include. As the MS1 scoring procedure will not preserve the
                # MS2 mapping, we must keep a mapping from Chromatogram Key to mapped tandem
                # matches to re-align later
                kept_annotated_matches = []
                key_to_tandem = defaultdict(list)
                for match in annotated_matches:
                    accepted = False
                    best_score = 0
                    key_to_tandem[match.key].extend(match.tandem_solutions)
                    for gsm in match.tandem_solutions:
                        if gsm.score > best_score:
                            best_score = gsm.score
                        if gsm.score > self.require_msms_signature:
                            accepted = True
                            break
                    if accepted:
                        kept_annotated_matches.append(match)
                    else:
                        self.debug(
                            "%s was discarded with insufficient MS/MS evidence %f" % (
                                match, best_score))
                kept_annotated_matches = ChromatogramFilter(kept_annotated_matches)
                processor.evaluate_chromatograms(kept_annotated_matches)
                for solution in processor.solutions:
                    mapped = []
                    try:
                        gsms = key_to_tandem[solution.key]
                        for gsm in gsms:
                            if solution.spans_time_point(gsm.scan_time):
                                mapped.append(gsm)
                        solution.tandem_solutions = mapped
                    except KeyError:
                        solution.tandem_solutions = []
                        continue
                processor.solutions = ChromatogramFilter([
                    solution for solution in processor.solutions
                    if len(solution.tandem_solutions) > 0
                ])
                processor.accepted_solutions = ChromatogramFilter([
                    solution for solution in processor.accepted_solutions
                    if len(solution.tandem_solutions) > 0
                ])
        else:
            processor.run()

    def run(self):
        peak_loader = self.make_peak_loader()
        database = self.make_database()
        smallest_database_mass = database.lowest_mass
        minimum_mass_shift = min([m.mass for m in self.mass_shifts]) if self.mass_shifts else 0
        if minimum_mass_shift < 0:
            smallest_database_mass = smallest_database_mass + minimum_mass_shift
        if smallest_database_mass > self.minimum_mass:
            self.log("The smallest possible database mass is %f, raising the minimum mass to extract." % (
                smallest_database_mass))
            self.minimum_mass = smallest_database_mass

        extractor = self.make_chromatogram_extractor(peak_loader)
        proc = self.make_chromatogram_processor(extractor, database)
        self.processor = proc
        self.process_chromatograms(proc, peak_loader, database)
        self.save_solutions(proc.solutions, extractor, database, proc.evaluator)
        return proc


class MzMLGlycanChromatogramAnalyzer(GlycanChromatogramAnalyzer):
    def __init__(self, database_connection, hypothesis_id, sample_path, output_path,
                 mass_shifts=None, mass_error_tolerance=1e-5, grouping_error_tolerance=1.5e-5,
                 scoring_model=None, minimum_mass=500., regularize=None,
                 regularization_model=None, network=None, analysis_name=None, delta_rt=0.5,
                 require_msms_signature=0, msn_mass_error_tolerance=2e-5,
                 n_processes=4):
        super(MzMLGlycanChromatogramAnalyzer, self).__init__(
            database_connection, hypothesis_id, -1, mass_shifts,
            mass_error_tolerance, grouping_error_tolerance,
            scoring_model, minimum_mass, regularize, regularization_model, network,
            analysis_name, delta_rt, require_msms_signature, msn_mass_error_tolerance,
            n_processes)
        self.sample_path = sample_path
        self.output_path = output_path

    def make_peak_loader(self):
        peak_loader = ProcessedMzMLDeserializer(self.sample_path)
        if peak_loader.extended_index is None:
            if not peak_loader.has_index_file():
                self.log("Index file missing. Rebuilding.")
                peak_loader.build_extended_index()
            else:
                peak_loader.read_index_file()
            if peak_loader.extended_index is None or len(peak_loader.extended_index.ms1_ids) < 1:
                raise ValueError("Sample Data Invalid: Could not validate MS Index")

        return peak_loader

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [ScanStub(o, peak_loader)
                      for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def save_solutions(self, solutions, extractor, database, evaluator):
        if self.analysis_name is None or self.output_path is None:
            return
        self.log('Saving solutions')

        exporter = GlycanCompositionChromatogramAnalysisSerializer(
            self.output_path, self.analysis_name, extractor.peak_loader.sample_run,
            solutions, database, extractor)

        param_dict = {
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_path,
            "sample_path": os.path.abspath(self.sample_path),
            "sample_name": extractor.peak_loader.sample_run.name,
            "sample_uuid": extractor.peak_loader.sample_run.uuid,
            "mass_error_tolerance": self.mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "mass_shifts": [mass_shift.name for mass_shift in self.mass_shifts],
            "minimum_mass": self.minimum_mass,
            "require_msms_signature": self.require_msms_signature,
            "msn_mass_error_tolerance": self.msn_mass_error_tolerance
        }

        evaluator.update_parameters(param_dict)

        exporter.run()
        exporter.set_parameters(param_dict)
        self.analysis = exporter.analysis
        self.analysis_id = exporter.analysis.id


class GlycopeptideSearchStrategy(Enum):
    target_internal_decoy_competition = "target-internal-decoy-competition"
    target_decoy_competition = "target-decoy-competition"
    multipart_target_decoy_competition = "multipart-target-decoy-competition"


class GlycopeptideLCMSMSAnalyzer(TaskBase):
    def __init__(self, database_connection, hypothesis_id, sample_run_id,
                 analysis_name=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                 tandem_scoring_model=None, minimum_mass=1000., save_unidentified=False,
                 oxonium_threshold=0.05, scan_transformer=None, mass_shifts=None, n_processes=5,
                 spectrum_batch_size=1000, use_peptide_mass_filter=False, maximum_mass=float('inf'),
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 permute_decoy_glycans=False, rare_signatures=False):
        if tandem_scoring_model is None:
            tandem_scoring_model = CoverageWeightedBinomialScorer
        if peak_shape_scoring_model is None:
            peak_shape_scoring_model = GeneralScorer.clone()
            peak_shape_scoring_model.add_feature(get_feature("null_charge"))
        if scan_transformer is None:
            def scan_transformer(x): # pylint: disable=function-redefined
                return x
        if mass_shifts is None:
            mass_shifts = []
        if Unmodified not in mass_shifts:
            mass_shifts = [Unmodified] + list(mass_shifts)
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
        self.mass_shifts = mass_shifts
        self.analysis = None
        self.analysis_id = None
        self.minimum_mass = minimum_mass
        self.save_unidentified = save_unidentified
        self.minimum_oxonium_ratio = oxonium_threshold
        self.scan_transformer = scan_transformer
        self.n_processes = n_processes
        self.spectrum_batch_size = spectrum_batch_size
        self.use_peptide_mass_filter = use_peptide_mass_filter
        self.maximum_mass = maximum_mass
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits
        self.file_manager = TempFileManager()
        self.fdr_estimator = None
        self.permute_decoy_glycans = permute_decoy_glycans
        self.rare_signatures = rare_signatures

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

    def prepare_cache_seeds(self, database):

        def flatten(nested):
            return set([y for x in nested for y in x])

        # gcs = database.query(serialize.GlycanCombination).all()
        # gcs = [
        #     (
        #         gc.convert(),
        #         # flatten(gc.component_classes)
        #         set()
        #     )
        #     for gc in gcs
        # ]

        # oxonium_ion_cache.populate([gc for gc, tps in gcs])
        # CachingStubGlycopeptideStrategy.populate([gc for gc, tps in gcs if "N-Glycan" in tps])
        seeds = {
            # "oxonium_ion_cache": pickle.dumps(oxonium_ion_cache, -1),
            # "n_glycan_stub_cache": pickle.dumps(CachingStubGlycopeptideStrategy.get_cache(), -1)
        }
        return seeds

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [o.product for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def make_search_engine(self, msms_scans, database, peak_loader):
        searcher = GlycopeptideDatabaseSearchIdentifier(
            [scan for scan in msms_scans
             if scan.precursor_information.neutral_mass < self.maximum_mass],
            self.tandem_scoring_model, database,
            peak_loader.convert_scan_id_to_retention_time,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            scan_transformer=self.scan_transformer,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            use_peptide_mass_filter=self.use_peptide_mass_filter,
            file_manager=self.file_manager,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            permute_decoy_glycans=self.permute_decoy_glycans)
        return searcher

    def do_search(self, searcher):
        assigned_spectra = searcher.search(
            precursor_error_tolerance=self.mass_error_tolerance,
            error_tolerance=self.msn_mass_error_tolerance,
            batch_size=self.spectrum_batch_size,
            rare_signatures=self.rare_signatures)
        return assigned_spectra

    def estimate_fdr(self, searcher, target_decoy_set):
        return searcher.estimate_fdr(*target_decoy_set, decoy_pseudocount=0.0)

    def map_chromatograms(self, searcher, extractor, target_hits):
        """Map identified spectrum matches onto extracted chromatogram groups, selecting the
        best overall structure for each chromatogram group and merging disjoint chromatograms
        which are assigned the same structure.

        The structure assigned to a chromatogram group is not necessarily the only structure that
        is reasonable and there may be ambiguity. When the same exact structure with different source
        information (duplicate peptide sequences, common subsequence between proteins) is assigned to
        a chromatogram group, every alternative structure is included as well.

        Parameters
        ----------
        searcher : object
            The search algorithm implementation providing a `map_to_chromatograms` method
        extractor : :class:`~.ChromatogramExtractor` or Iterable of :class:`Chromatogram`
            The chromatograms to map to
        target_hits : list
            The list of target spectrum matches

        Returns
        -------
        merged: :class:`~.ChromatogramFilter`
            The chromatograms after all structure assignment, aggregation and merging
            is complete.
        orphans: list
            The set of spectrum matches which were not assigned to a chromatogram.
        """

        def threshold_fn(x):
            return x.q_value <= self.psm_fdr_threshold

        chromatograms = tuple(extractor)

        chroma_with_sols, orphans = searcher.map_to_chromatograms(
            chromatograms, target_hits, self.mass_error_tolerance,
            threshold_fn=threshold_fn)

        self.log("Aggregating Assigned Entities")
        merged = chromatogram_mapping.aggregate_by_assigned_entity(
            chroma_with_sols, threshold_fn=threshold_fn)

        return merged, orphans

    def score_chromatograms(self, merged):
        """Calculate the MS1 score for each chromatogram.

        Parameters
        ----------
        merged : Iterable of :class:`Chromatogram`
            The chromatograms to score

        Returns
        -------
        list:
            The scored chromatograms
        """
        chroma_scoring_model = self.peak_shape_scoring_model
        scored_merged = []
        n = len(merged)
        i = 0
        for c in merged:
            i += 1
            if i % 500 == 0:
                self.log("%0.2f%% chromatograms evaluated (%d/%d) %r" % (i * 100. / n, i, n, c))
            try:
                chrom_score = chroma_scoring_model.logitscore(c)
                scored_merged.append(ChromatogramSolution(
                    c, scorer=chroma_scoring_model, score=chrom_score))
            except (IndexError, ValueError) as e:
                self.log("Could not score chromatogram %r due to %s" % (c, e))
                scored_merged.append(ChromatogramSolution(c, score=0.0))
        return scored_merged

    def assign_consensus(self, scored_merged, orphans):
        self.log("Assigning consensus glycopeptides to spectrum clusters")
        assigned_list = list(scored_merged)
        assigned_list.extend(orphans)
        gps, unassigned = identified_glycopeptide.extract_identified_structures(
            assigned_list, lambda x: x.q_value <= self.psm_fdr_threshold)
        return gps, unassigned

    def rank_target_hits(self, searcher, target_decoy_set):
        '''Estimate the FDR using the searcher's method, and
        count the number of acceptable target matches. Return
        the full set of target matches for downstream use.

        Parameters
        ----------
        searcher: object
            The search algorithm implementation, providing an `estimate_fdr` method
        target_decoy_set: TargetDecoySet

        Returns
        -------
        Iterable of SpectrumMatch-like objects
        '''
        self.log("Estimating FDR")
        tda = self.estimate_fdr(searcher, target_decoy_set)
        if tda is not None:
            tda.pack()
        self.fdr_estimator = tda
        target_hits = target_decoy_set.target_matches
        n_below = 0
        for target in target_hits:
            if target.q_value <= self.psm_fdr_threshold:
                n_below += 1
        self.log("%d spectrum matches accepted" % (n_below,))
        return target_hits

    def run(self):
        peak_loader = self.make_peak_loader()
        database = self.make_database()
        extractor = self.make_chromatogram_extractor(peak_loader)

        self.log("Loading MS/MS")

        msms_scans = self.load_msms(peak_loader)

        # Traditional LC-MS/MS Database Search
        searcher = self.make_search_engine(msms_scans, database, peak_loader)
        target_decoy_set = self.do_search(searcher)

        self.target_hit_count = target_decoy_set.target_count()
        self.decoy_hit_count = target_decoy_set.decoy_count()

        if target_decoy_set.target_count() == 0:
            self.log("No target matches were found.")
            return [], [], TargetDecoySet([], [])

        target_hits = self.rank_target_hits(searcher, target_decoy_set)

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
        return gps, unassigned, target_decoy_set

    def _filter_out_poor_matches_before_saving(self, identified_glycopeptides):
        filt = QValueRetentionStrategy(max(0.75, self.psm_fdr_threshold * 10.))
        for idgp in identified_glycopeptides:
            for gpsm_set in idgp.tandem_solutions:
                gpsm_set.threshold(filt)
        return identified_glycopeptides

    def _build_analysis_saved_parameters(self, identified_glycopeptides, unassigned_chromatograms,
                                         chromatogram_extractor, database):
        return {
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "mass_error_tolerance": self.mass_error_tolerance,
            "fragment_error_tolerance": self.msn_mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "psm_fdr_threshold": self.psm_fdr_threshold,
            "minimum_mass": self.minimum_mass,
            "use_peptide_mass_filter": self.use_peptide_mass_filter,
            "mass_shifts": self.mass_shifts,
            "maximum_mass": self.maximum_mass,
            "fdr_estimator": self.fdr_estimator,
            "permute_decoy_glycans": self.permute_decoy_glycans,
            "rare_signatures": self.rare_signatures,
        }

    def save_solutions(self, identified_glycopeptides, unassigned_chromatograms,
                       chromatogram_extractor, database):
        if self.analysis_name is None:
            return
        self.log("Saving Results To \"%s\"" % (self.database_connection,))
        analysis_saver = AnalysisSerializer(self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(chromatogram_extractor.peak_mapping)
        analysis_saver.set_analysis_type(AnalysisTypeEnum.glycopeptide_lc_msms.name)
        analysis_saver.set_parameters(
            self._build_analysis_saved_parameters(
                identified_glycopeptides, unassigned_chromatograms,
                chromatogram_extractor, database))

        analysis_saver.save_glycopeptide_identification_set(
            self._filter_out_poor_matches_before_saving(identified_glycopeptides))
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
        for path in self.file_manager.dir():
            if os.path.isdir(path):
                continue
            self.analysis.add_file(path, compress=True)
        analysis_saver.session.add(self.analysis)
        analysis_saver.commit()


class MzMLGlycopeptideLCMSMSAnalyzer(GlycopeptideLCMSMSAnalyzer):
    def __init__(self, database_connection, hypothesis_id, sample_path, output_path,
                 analysis_name=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                 tandem_scoring_model=None, minimum_mass=1000., save_unidentified=False,
                 oxonium_threshold=0.05, scan_transformer=None, mass_shifts=None,
                 n_processes=5, spectrum_batch_size=1000, use_peptide_mass_filter=False,
                 maximum_mass=float('inf'), probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True, permute_decoy_glycans=False, rare_signatures=False):
        super(MzMLGlycopeptideLCMSMSAnalyzer, self).__init__(
            database_connection,
            hypothesis_id, -1,
            analysis_name, grouping_error_tolerance,
            mass_error_tolerance, msn_mass_error_tolerance,
            psm_fdr_threshold, peak_shape_scoring_model,
            tandem_scoring_model, minimum_mass,
            save_unidentified, oxonium_threshold,
            scan_transformer, mass_shifts, n_processes, spectrum_batch_size,
            use_peptide_mass_filter, maximum_mass,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            trust_precursor_fits=trust_precursor_fits, permute_decoy_glycans=permute_decoy_glycans,
            rare_signatures=rare_signatures)
        self.sample_path = sample_path
        self.output_path = output_path

    def make_peak_loader(self):
        peak_loader = ProcessedMzMLDeserializer(self.sample_path)
        if peak_loader.extended_index is None:
            if not peak_loader.has_index_file():
                self.log("Index file missing. Rebuilding.")
                peak_loader.build_extended_index()
            else:
                peak_loader.read_index_file()
            if peak_loader.extended_index is None or len(peak_loader.extended_index.msn_ids) < 1:
                raise ValueError("Sample Data Invalid: Could not validate MS/MS Index")
        return peak_loader

    def load_msms(self, peak_loader):
        prec_info = peak_loader.precursor_information()
        msms_scans = [ScanStub(o, peak_loader) for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def _build_analysis_saved_parameters(self, identified_glycopeptides, unassigned_chromatograms,
                                         chromatogram_extractor, database):
        return {
            "hypothesis_id": self.hypothesis_id,
            "sample_run_id": self.sample_run_id,
            "sample_path": os.path.abspath(self.sample_path),
            "sample_name": chromatogram_extractor.peak_loader.sample_run.name,
            "sample_uuid": chromatogram_extractor.peak_loader.sample_run.uuid,
            "mass_error_tolerance": self.mass_error_tolerance,
            "fragment_error_tolerance": self.msn_mass_error_tolerance,
            "grouping_error_tolerance": self.grouping_error_tolerance,
            "psm_fdr_threshold": self.psm_fdr_threshold,
            "minimum_mass": self.minimum_mass,
            "maximum_mass": self.maximum_mass,
            "mass_shifts": self.mass_shifts,
            "use_peptide_mass_filter": self.use_peptide_mass_filter,
            "database": str(self.database_connection),
            "search_strategy": 'target-internal-decoy-competition',
            "trust_precursor_fits": self.trust_precursor_fits,
            "probing_range_for_missing_precursors": self.probing_range_for_missing_precursors,
            "scoring_model": self.peak_shape_scoring_model,
            "fdr_estimator": self.fdr_estimator,
            "permute_decoy_glycans": self.permute_decoy_glycans,
            "tandem_scoring_model": self.tandem_scoring_model,
            "rare_signatures": self.rare_signatures,
        }

    def make_analysis_serializer(self, output_path, analysis_name, sample_run, identified_glycopeptides,
                                 unassigned_chromatograms, database, chromatogram_extractor):
        return GlycopeptideMSMSAnalysisSerializer(
            output_path, analysis_name, sample_run, identified_glycopeptides,
            unassigned_chromatograms, database, chromatogram_extractor)

    def save_solutions(self, identified_glycopeptides, unassigned_chromatograms,
                       chromatogram_extractor, database):
        if self.analysis_name is None:
            return
        self.log("Saving Results To \"%s\"" % (self.output_path,))
        exporter = self.make_analysis_serializer(
            self.output_path,
            self.analysis_name,
            chromatogram_extractor.peak_loader.sample_run,
            self._filter_out_poor_matches_before_saving(identified_glycopeptides),
            unassigned_chromatograms,
            database,
            chromatogram_extractor)

        exporter.run()

        exporter.set_parameters(
            self._build_analysis_saved_parameters(
                identified_glycopeptides, unassigned_chromatograms,
                chromatogram_extractor, database))

        self.analysis = exporter.analysis
        self.analysis_id = exporter.analysis.id
        for path in self.file_manager.dir():
            if os.path.isdir(path):
                continue
            self.analysis.add_file(path, compress=True)

        session = object_session(self.analysis)
        session.add(self.analysis)
        session.commit()


class MzMLComparisonGlycopeptideLCMSMSAnalyzer(MzMLGlycopeptideLCMSMSAnalyzer):
    def __init__(self, database_connection, decoy_database_connection, hypothesis_id,
                 sample_path, output_path,
                 analysis_name=None, grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05, peak_shape_scoring_model=None,
                 tandem_scoring_model=None, minimum_mass=1000., save_unidentified=False,
                 oxonium_threshold=0.05, scan_transformer=None, mass_shifts=None,
                 n_processes=5, spectrum_batch_size=1000, use_peptide_mass_filter=False,
                 maximum_mass=float('inf'), use_decoy_correction_threshold=None,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 permute_decoy_glycans=False, rare_signatures=False):
        if use_decoy_correction_threshold is None:
            use_decoy_correction_threshold = 0.33
        if tandem_scoring_model == CoverageWeightedBinomialScorer:
            tandem_scoring_model = CoverageWeightedBinomialModelTree
        super(MzMLComparisonGlycopeptideLCMSMSAnalyzer, self).__init__(
            database_connection, hypothesis_id, sample_path, output_path,
            analysis_name, grouping_error_tolerance, mass_error_tolerance,
            msn_mass_error_tolerance, psm_fdr_threshold, peak_shape_scoring_model,
            tandem_scoring_model, minimum_mass, save_unidentified,
            oxonium_threshold, scan_transformer, mass_shifts,
            n_processes, spectrum_batch_size, use_peptide_mass_filter,
            maximum_mass, probing_range_for_missing_precursors,
            trust_precursor_fits, permute_decoy_glycans=permute_decoy_glycans,
            rare_signatures=rare_signatures)
        self.decoy_database_connection = decoy_database_connection
        self.use_decoy_correction_threshold = use_decoy_correction_threshold

    def make_search_engine(self, msms_scans, database, peak_loader):
        searcher = ExclusiveGlycopeptideDatabaseSearchComparer(
            [scan for scan in msms_scans
             if scan.precursor_information.neutral_mass < self.maximum_mass],
            self.tandem_scoring_model, database, self.make_decoy_database(),
            peak_loader.convert_scan_id_to_retention_time,
            minimum_oxonium_ratio=self.minimum_oxonium_ratio,
            scan_transformer=self.scan_transformer,
            n_processes=self.n_processes,
            mass_shifts=self.mass_shifts,
            use_peptide_mass_filter=self.use_peptide_mass_filter,
            file_manager=self.file_manager,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            permute_decoy_glycans=self.permute_decoy_glycans)
        return searcher

    def make_decoy_database(self):
        database = GlycopeptideDiskBackedStructureDatabase(
            self.decoy_database_connection, self.hypothesis_id)
        return database

    def _build_analysis_saved_parameters(self, identified_glycopeptides, unassigned_chromatograms,
                                         chromatogram_extractor, database):
        result = super(MzMLComparisonGlycopeptideLCMSMSAnalyzer, self)._build_analysis_saved_parameters(
            identified_glycopeptides, unassigned_chromatograms,
            chromatogram_extractor, database)
        target_database = database
        decoy_database = self.make_decoy_database()
        result.update({
            "target_database": str(self.database_connection),
            "target_database_size": len(target_database),
            "decoy_database_size": len(decoy_database),
            "decoy_database": str(self.decoy_database_connection),
            "decoy_correction_threshold": self.use_decoy_correction_threshold,
            "search_strategy": 'target-decoy-competition',
        })
        return result

    def estimate_fdr(self, searcher, target_decoy_set):
        if target_decoy_set.decoy_count() / float(
                target_decoy_set.target_count()) < self.use_decoy_correction_threshold:
            targets_per_decoy = 0.5
            decoy_correction = 1
        else:
            targets_per_decoy = 1.0
            decoy_correction = 0
        return searcher.estimate_fdr(
            *target_decoy_set, decoy_correction=decoy_correction,
            target_weight=targets_per_decoy)


class MultipartGlycopeptideLCMSMSAnalyzer(MzMLGlycopeptideLCMSMSAnalyzer):
    def __init__(self, database_connection, decoy_database_connection, target_hypothesis_id,
                 decoy_hypothesis_id, sample_path, output_path, analysis_name=None,
                 grouping_error_tolerance=1.5e-5, mass_error_tolerance=1e-5,
                 msn_mass_error_tolerance=2e-5, psm_fdr_threshold=0.05,
                 peak_shape_scoring_model=None, tandem_scoring_model=LogIntensityScorer,
                 minimum_mass=1000., save_unidentified=False,
                 glycan_score_threshold=1, mass_shifts=None,
                 n_processes=5, spectrum_batch_size=100,
                 maximum_mass=float('inf'), probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True, use_memory_database=True,
                 fdr_estimation_strategy=None, glycosylation_site_models_path=None,
                 permute_decoy_glycans=False, fragile_fucose=False, rare_signatures=False):
        if tandem_scoring_model == CoverageWeightedBinomialScorer:
            tandem_scoring_model = CoverageWeightedBinomialModelTree
        if fdr_estimation_strategy is None:
            fdr_estimation_strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
        super(MultipartGlycopeptideLCMSMSAnalyzer, self).__init__(
            database_connection, target_hypothesis_id, sample_path, output_path,
            analysis_name, grouping_error_tolerance, mass_error_tolerance,
            msn_mass_error_tolerance, psm_fdr_threshold, peak_shape_scoring_model,
            tandem_scoring_model, minimum_mass, save_unidentified,
            0, None, mass_shifts,
            n_processes, spectrum_batch_size, True,
            maximum_mass, probing_range_for_missing_precursors,
            trust_precursor_fits,
            # The multipart scoring algorithm automatically implies permute_decoy_glycan
            # fragment masses.
            permute_decoy_glycans=True)
        self.fragile_fucose = fragile_fucose
        self.rare_signatures = rare_signatures
        self.glycan_score_threshold = glycan_score_threshold
        self.decoy_database_connection = decoy_database_connection
        self.use_memory_database = use_memory_database
        self.decoy_hypothesis_id = decoy_hypothesis_id
        self.fdr_estimation_strategy = fdr_estimation_strategy
        self.glycosylation_site_models_path = glycosylation_site_models_path
        self.fdr_estimator = None
        self.precursor_mass_error_distribution = None

    @property
    def target_hypothesis_id(self):
        return self.hypothesis_id

    @target_hypothesis_id.setter
    def target_hypothesis_id(self, value):
        self.hypothesis_id = value

    def make_database(self):
        if self.use_memory_database:
            database = MultipartGlycopeptideIdentifier.build_default_memory_backed_db_wrapper(
                self.database_connection, hypothesis_id=self.target_hypothesis_id)
        else:
            database = MultipartGlycopeptideIdentifier.build_default_disk_backed_db_wrapper(
                self.database_connection, hypothesis_id=self.target_hypothesis_id)
        return database

    def make_decoy_database(self):
        if self.use_memory_database:
            database = MultipartGlycopeptideIdentifier.build_default_memory_backed_db_wrapper(
                self.decoy_database_connection, hypothesis_id=self.decoy_hypothesis_id)
        else:
            database = MultipartGlycopeptideIdentifier.build_default_disk_backed_db_wrapper(
                self.decoy_database_connection, hypothesis_id=self.decoy_hypothesis_id)
        return database

    def make_search_engine(self, msms_scans, database, peak_loader):
        cache_seeds = self.prepare_cache_seeds(
            serialize.DatabaseBoundOperation(self.database_connection))
        searcher = MultipartGlycopeptideIdentifier(
            [scan for scan in msms_scans
             if scan.precursor_information.neutral_mass < self.maximum_mass],
            self.tandem_scoring_model, database, self.make_decoy_database(),
            peak_loader,
            mass_shifts=self.mass_shifts,
            glycan_score_threshold=self.glycan_score_threshold,
            n_processes=self.n_processes,
            file_manager=self.file_manager,
            probing_range_for_missing_precursors=self.probing_range_for_missing_precursors,
            trust_precursor_fits=self.trust_precursor_fits,
            fdr_estimation_strategy=self.fdr_estimation_strategy,
            glycosylation_site_models_path=self.glycosylation_site_models_path,
            cache_seeds=cache_seeds, evaluation_kwargs={
                "fragile_fucose": self.fragile_fucose,
                "rare_signatures": self.rare_signatures,
            })
        return searcher

    def estimate_fdr(self, searcher, target_decoy_set):
        return searcher.estimate_fdr(target_decoy_set)

    def rank_target_hits(self, searcher, target_decoy_set):
        '''Estimate the FDR using the searcher's method, and
        count the number of acceptable target matches. Return
        the full set of target matches for downstream use.

        Parameters
        ----------
        searcher: object
            The search algorithm implementation, providing an `estimate_fdr` method
        target_decoy_set: TargetDecoySet

        Returns
        -------
        Iterable of SpectrumMatch-like objects
        '''
        self.log("Estimating FDR")
        _groups, fdr_estimator = self.estimate_fdr(searcher, target_decoy_set)
        self.fdr_estimator = fdr_estimator
        if self.fdr_estimator is not None:
            self.fdr_estimator.pack()
        target_hits = target_decoy_set.target_matches
        n_below = 0
        for target in target_hits:
            if target.q_value <= self.psm_fdr_threshold:
                n_below += 1
        self.log("%d spectrum matches accepted" % (n_below,))
        return target_hits

    def make_analysis_serializer(self, output_path, analysis_name, sample_run, identified_glycopeptides,
                                 unassigned_chromatograms, database, chromatogram_extractor):
        return DynamicGlycopeptideMSMSAnalysisSerializer(
            output_path, analysis_name, sample_run,
            self._filter_out_poor_matches_before_saving(identified_glycopeptides),
            unassigned_chromatograms, database, chromatogram_extractor)

    def _build_analysis_saved_parameters(self, identified_glycopeptides, unassigned_chromatograms,
                                         chromatogram_extractor, database):
        database = GlycopeptideDiskBackedStructureDatabase(self.database_connection)
        result = super(MultipartGlycopeptideLCMSMSAnalyzer, self)._build_analysis_saved_parameters(
            identified_glycopeptides, unassigned_chromatograms,
            chromatogram_extractor, database)
        result.update({
            "target_database": str(self.database_connection),
            "decoy_database": str(self.decoy_database_connection),
            "search_strategy": 'multipart-target-decoy-competition',
            "fdr_estimation_strategy": self.fdr_estimation_strategy,
            "fdr_estimator": self.fdr_estimator,
            "fragile_fucose": self.fragile_fucose,
            "rare_signatures": self.rare_signatures,
            "glycosylation_site_models_path": self.glycosylation_site_models_path,
        })
        return result
