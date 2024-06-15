'''
High level analytical pipeline implementations.

Each class is designed to encapsulate a single broad task, i.e.
LC-MS/MS deconvolution or structure identification
'''
import os
import platform

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type, Sequence
from glycresoft.chromatogram_tree.mass_shift import MassShiftBase
from glycresoft.tandem.glycopeptide.dynamic_generation.journal import SolutionSetGrouper

import numpy as np

from ms_deisotope.output import ProcessedMSFileLoader
from ms_deisotope.data_source import ProcessedRandomAccessScanSource
from ms_deisotope.tools.deisotoper.workflow import SampleConsumer as _SampleConsumer

import glypy
from glypy.utils import Enum

from glycopeptidepy.utils.collectiontools import descending_combination_counter
from glycopeptidepy.structure.modification import rule_string_to_specialized_rule

from glycresoft.database.disk_backed_database import (
    DiskBackedStructureDatabaseBase,
    GlycanCompositionDiskBackedStructureDatabase,
    GlycopeptideDiskBackedStructureDatabase)

from glycresoft.database.analysis import (
    GlycanCompositionChromatogramAnalysisSerializer,
    DynamicGlycopeptideMSMSAnalysisSerializer,
    GlycopeptideMSMSAnalysisSerializer)

from glycresoft.serialize import (
    DatabaseScanDeserializer, AnalysisSerializer,
    AnalysisTypeEnum, object_session)

from glycresoft.scoring import (
    ChromatogramSolution)
from glycresoft.tandem.target_decoy.base import TargetDecoyAnalyzer

from glycresoft.trace import (
    ChromatogramExtractor,
    LogitSumChromatogramProcessor,
    LaplacianRegularizedChromatogramProcessor,
    ChromatogramEvaluator)


from glycresoft.chromatogram_tree import ChromatogramFilter, SimpleChromatogram, Unmodified

from glycresoft.models import GeneralScorer, get_feature

from glycresoft.scoring.elution_time_grouping import (
    GlycopeptideChromatogramProxy, GlycoformAggregator,
    GlycopeptideElutionTimeModelBuildingPipeline,
    PeptideYUtilizationPreservingRevisionValidator,
    OxoniumIonRequiringUtilizationRevisionValidator,
    CompoundRevisionValidator, ModelEnsemble as GlycopeptideElutionTimeModelEnsemble)

from glycresoft.structure import ScanStub, ScanInformation

from glycresoft.piped_deconvolve import ScanGenerator

from glycresoft.tandem.chromatogram_mapping import (
    SpectrumMatchUpdater,
    AnnotatedChromatogramAggregator,
    TandemAnnotatedChromatogram,
    TandemSolutionsWithoutChromatogram,
    MS2RevisionValidator,
    SignalUtilizationMS2RevisionValidator,
    RevisionSummary
)

from glycresoft.tandem import chromatogram_mapping
from glycresoft.tandem.target_decoy import TargetDecoySet
from glycresoft.tandem.temp_store import TempFileManager
from glycresoft.tandem.workflow import SearchEngineBase

from glycresoft.tandem.spectrum_match.spectrum_match import MultiScoreSpectrumMatch, SpectrumMatch
from glycresoft.tandem.spectrum_match.solution_set import QValueRetentionStrategy, SpectrumSolutionSet

from glycresoft.tandem.glycopeptide.scoring import (
    LogIntensityScorer,
    CoverageWeightedBinomialScorer,
    CoverageWeightedBinomialModelTree)

from glycresoft.tandem.glycopeptide.glycopeptide_matcher import (
    GlycopeptideDatabaseSearchIdentifier,
    ExclusiveGlycopeptideDatabaseSearchComparer)

from glycresoft.tandem.glycopeptide.dynamic_generation import (
    MultipartGlycopeptideIdentifier, GlycopeptideFDREstimationStrategy, IdKeyMaker)

from glycresoft.tandem.glycopeptide import (
    identified_structure as identified_glycopeptide)

from glycresoft.tandem.glycopeptide.identified_structure import IdentifiedGlycopeptide

from glycresoft.tandem.glycan.composition_matching import SignatureIonMapper
from glycresoft.tandem.glycan.scoring.signature_ion_scoring import SignatureIonScorer

from glycresoft.tandem.localize import EvaluatedSolutionBins, ScanLoadingModificationLocalizationSearcher

from glycresoft.task import TaskBase
from glycresoft import serialize

from glycresoft.tandem.coelute import CoElutionAdductFinder


debug_mode = bool(os.environ.get('GLYCRESOFTDEBUG', False))


class SampleConsumer(TaskBase, _SampleConsumer):
    scan_generator_cls = ScanGenerator


class ChromatogramSummarizer(TaskBase):
    """
    Implement the simple diagnostic chromatogram extraction pipeline.

    Given a deconvoluted mzML file produced by :class:`SampleConsumer` this will build
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
        """Create a reader for the deconvoluted LC-MS data file."""
        scan_loader = ProcessedMSFileLoader(self.mzml_path)
        return scan_loader

    def estimate_intensity_threshold(self, scan_loader):
        """Build the MS1 peak intensity distribution to estimate the global intensity threshold.

        This threshold is then used when extracting chromatograms.
        """
        acc = []
        for scan_id in scan_loader.extended_index.ms1_ids:
            header = scan_loader.get_scan_header_by_id(scan_id)
            acc.extend(header.arrays.intensity)
        self.intensity_threshold = np.percentile(acc, self.threshold_percentile)
        return self.intensity_threshold

    def extract_chromatograms(self, scan_loader):
        """Perform the chromatogram extraction process."""
        extractor = ChromatogramExtractor(
            scan_loader, minimum_intensity=self.intensity_threshold,
            minimum_mass=self.minimum_mass)
        chroma = extractor.run()
        return chroma, extractor.total_ion_chromatogram, extractor.base_peak_chromatogram

    def extract_signature_ion_traces(self, scan_loader):
        """Skim the MSn spectra to look for oxonium ion signatures over time."""
        from glycresoft.tandem.oxonium_ions import standard_oxonium_ions
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
    """Analyze glycan LC-MS profiling data, assigning glycan compositions to extracted chromatograms.

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
        """Map MSn scans to chromatograms matched by precursor mass, and evaluate each glycan compostion-spectrum match.

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
        peak_loader = ProcessedMSFileLoader(self.sample_path)
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
                 permute_decoy_glycans=False, rare_signatures=False, model_retention_time=False,
                 evaluation_kwargs=None):
        if evaluation_kwargs is None:
            evaluation_kwargs = {}
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
        self.analysis = None
        self.analysis_id = None

        self.mass_error_tolerance = mass_error_tolerance
        self.msn_mass_error_tolerance = msn_mass_error_tolerance
        self.grouping_error_tolerance = grouping_error_tolerance
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits
        self.minimum_mass = minimum_mass
        self.maximum_mass = maximum_mass
        self.minimum_oxonium_ratio = oxonium_threshold
        self.use_peptide_mass_filter = use_peptide_mass_filter

        self.peak_shape_scoring_model = peak_shape_scoring_model
        self.psm_fdr_threshold = psm_fdr_threshold
        self.tandem_scoring_model = tandem_scoring_model
        self.fdr_estimator = None
        self.extra_msn_evaluation_kwargs = evaluation_kwargs
        self.retention_time_model = None

        self.mass_shifts = mass_shifts

        self.rare_signatures = rare_signatures
        self.model_retention_time = model_retention_time
        self.permute_decoy_glycans = permute_decoy_glycans
        self.save_unidentified = save_unidentified
        self.spectrum_batch_size = spectrum_batch_size
        self.scan_transformer = scan_transformer

        self.n_processes = n_processes
        self.file_manager = TempFileManager()
        self.analysis_metadata = {
            "host_uname": platform.uname()._asdict()
        }

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
        seeds = {}
        return seeds

    def load_msms(self, peak_loader: ProcessedRandomAccessScanSource):
        prec_info = peak_loader.precursor_information()
        msms_scans = [o.product for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def make_msn_evaluation_kwargs(self):
        evaluation_kwargs = {
            "rare_signatures": self.rare_signatures,
        }
        evaluation_kwargs.update(self.extra_msn_evaluation_kwargs)
        return evaluation_kwargs

    def make_search_engine(self, msms_scans: List[ScanStub],
                           database,
                           peak_loader: ProcessedRandomAccessScanSource) -> SearchEngineBase:
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

    def do_search(self, searcher: SearchEngineBase) -> TargetDecoySet:
        kwargs = self.make_msn_evaluation_kwargs()
        assigned_spectra = searcher.search(
            precursor_error_tolerance=self.mass_error_tolerance,
            error_tolerance=self.msn_mass_error_tolerance,
            batch_size=self.spectrum_batch_size,
            **kwargs)
        return assigned_spectra

    def estimate_fdr(self, searcher: SearchEngineBase, target_decoy_set: TargetDecoySet) -> TargetDecoySet:
        return searcher.estimate_fdr(*target_decoy_set, decoy_pseudocount=0.0)

    def map_chromatograms(self,
                          searcher: SearchEngineBase,
                          extractor: ChromatogramExtractor,
                          target_hits: List[SpectrumSolutionSet],
                          revision_validator_type: Optional[Type[MS2RevisionValidator]]=None) -> Tuple[
                                                                                                    ChromatogramFilter,
                                                                                                    Sequence[TandemSolutionsWithoutChromatogram]
                                                                                                ]:
        """
        Map identified spectrum matches onto extracted chromatogram groups.

        It selects the best overall structure for each chromatogram group and merging disjoint
        chromatograms which are assigned the same structure.

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
        revision_validator_type : Type[MS2RevisionValidator]
            A type derived from :class:`~.MS2RevisionValidator` that

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

        if revision_validator_type is None:
            revision_validator_type = MS2RevisionValidator

        chromatograms = tuple(extractor)

        identified_spectrum_ids = {sset.scan.id for sset in target_hits}

        chroma_with_sols, orphans = searcher.map_to_chromatograms(
            chromatograms,
            target_hits,
            self.mass_error_tolerance,
            threshold_fn=threshold_fn
        )

        mapped_spectrum_ids = {sset.scan.id for chrom in chroma_with_sols for sset in chrom.tandem_solutions}
        orphan_spectrum_ids = {sset.scan.id for orph in orphans for sset in orph.tandem_solutions}

        mapping_lost_spectrum_ids = identified_spectrum_ids - (mapped_spectrum_ids | orphan_spectrum_ids)

        mapping_leaked_ssets = [
            sset for sset in target_hits
            if sset.scan.id in mapping_lost_spectrum_ids and
            threshold_fn(sset.best_solution())
        ]

        if mapping_leaked_ssets:
            self.log(f"Leaked {len(mapping_leaked_ssets)} spectra after mapping")
            if debug_mode:
                breakpoint()

        # Detect leaked spectrum matches between here and after aggregate_by_assigned_entity

        self.log("Aggregating Assigned Entities")
        merged = chromatogram_mapping.aggregate_by_assigned_entity(
            chroma_with_sols,
            threshold_fn=threshold_fn,
            revision_validator=revision_validator_type(threshold_fn),
        )

        aggregated_spectrum_ids = {sset.scan.id for chrom in merged for sset in chrom.tandem_solutions}
        aggregated_lost_spectrum_ids = identified_spectrum_ids - (aggregated_spectrum_ids | orphan_spectrum_ids)

        aggregated_leaked_ssets = [
            sset for sset in target_hits
            if sset.scan.id in aggregated_lost_spectrum_ids and
            threshold_fn(sset.best_solution())
        ]

        if aggregated_leaked_ssets:
            self.log(
                f"Leaked {len(aggregated_leaked_ssets)} spectra after aggregating")
            if debug_mode:
                breakpoint()

        return merged, orphans

    def score_chromatograms(self, merged: ChromatogramFilter):
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

    def assign_consensus(self, scored_merged: ChromatogramFilter,
                         orphans: Sequence[TandemSolutionsWithoutChromatogram]) -> Tuple[List[IdentifiedGlycopeptide],
                                                                                         List[ChromatogramSolution]]:
        self.log("Assigning consensus glycopeptides to spectrum clusters")
        assigned_list = list(scored_merged)
        assigned_list.extend(orphans)
        gps, unassigned = identified_glycopeptide.extract_identified_structures(
            assigned_list, lambda x: x.q_value <= self.psm_fdr_threshold)
        return gps, unassigned

    def rank_target_hits(self, searcher: SearchEngineBase,
                         target_decoy_set: TargetDecoySet) -> List[SpectrumSolutionSet]:
        """Estimate the FDR using the searcher's method.

        Also count the number of acceptable target matches. Return
        the full set of target matches for downstream use.

        Parameters
        ----------
        searcher: object
            The search algorithm implementation, providing an `estimate_fdr` method
        target_decoy_set: TargetDecoySet

        Returns
        -------
        Iterable of SpectrumMatch-like objects
        """
        self.log("Estimating FDR")
        tda: TargetDecoyAnalyzer = self.estimate_fdr(searcher, target_decoy_set)
        if tda is not None:
            tda.pack()
        self.fdr_estimator = tda
        target_hits = target_decoy_set.target_matches
        n_below = 0
        n_below_1 = 0
        n_below_5 = 0
        for target in target_hits:
            if target.q_value <= self.psm_fdr_threshold:
                n_below += 1
            if target.q_value <= 0.05:
                n_below_5 += 1
            if target.q_value <= 0.01:
                n_below_1 += 1
        self.log("%d spectrum matches accepted" % (n_below,))
        if self.psm_fdr_threshold != 0.05:
            self.log("%d spectra matched passing 5%% FDR" % n_below_5)
        if self.psm_fdr_threshold != 0.01:
            self.log("%d spectra matched passing 1%% FDR" % n_below_1)
        return target_hits

    def localize_modifications(self,
                               solution_sets: List[SpectrumSolutionSet],
                               scan_loader: ProcessedRandomAccessScanSource,
                               database: DiskBackedStructureDatabaseBase):
        pass

    def finalize_solutions(self, identifications: List[IdentifiedGlycopeptide]):
        pass

    def handle_adducts(self,
                       peak_loader: ProcessedRandomAccessScanSource,
                       identifications: List[IdentifiedGlycopeptide],
                       chromatograms: ChromatogramFilter,
                       mass_shifts: Optional[List[MassShiftBase]]=None) -> List[IdentifiedGlycopeptide]:
        if not mass_shifts:
            return identifications
        augmented_chromatograms = list(chromatograms)
        augmented_chromatograms.extend(identifications)
        augmented_chromatograms = ChromatogramFilter(augmented_chromatograms)

        msn_evaluation_kwargs = {"error_tolerance": self.msn_mass_error_tolerance}
        msn_evaluation_kwargs.update(self.extra_msn_evaluation_kwargs)

        scan_id_to_solution_set = {}
        for ident in identifications:
            for sset in ident.tandem_solutions:
                scan_id_to_solution_set[sset.scan.scan_id] = sset

        adduct_finder = CoElutionAdductFinder(
            scan_loader=peak_loader,
            chromatograms=augmented_chromatograms,
            msn_scoring_model=self.tandem_scoring_model,
            msn_evaluation_args=msn_evaluation_kwargs,
            fdr_estimator=self.fdr_estimator,
            scan_id_to_solution_set=scan_id_to_solution_set,
            threshold_fn=lambda x: x.q_value <= self.psm_fdr_threshold
        )

        for adduct in mass_shifts:
            self.log(f"... Handling {adduct.name}")
            identifications = adduct_finder.handle_adduct(
                identifications, adduct, self.mass_error_tolerance)

        return identifications

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
            self.save_solutions([], [], extractor, database)
            return [], [], TargetDecoySet([], [])

        target_hits = self.rank_target_hits(searcher, target_decoy_set)

        self.localize_modifications(target_hits, peak_loader, database)

        # Map MS/MS solutions to chromatograms.
        self.log("Building and Mapping Chromatograms")
        merged, orphans = self.map_chromatograms(searcher, extractor, target_hits)

        if not self.save_unidentified:
            merged = [chroma for chroma in merged if chroma.composition is not None]

        # Score chromatograms, both matched and unmatched
        self.log("Scoring chromatograms")
        scored_merged = self.score_chromatograms(merged)

        if self.model_retention_time and len(scored_merged) > 0:
            scored_merged, orphans = self.apply_retention_time_model(
                scored_merged, orphans, database, peak_loader)

        gps, unassigned = self.assign_consensus(scored_merged, orphans)

        self.finalize_solutions(gps)

        self.log("Saving solutions (%d identified glycopeptides)" % (len(gps),))
        self.save_solutions(gps, unassigned, extractor, database)
        return gps, unassigned, target_decoy_set

    def _get_glycan_compositions_from_database(self, database):
        glycan_query = database.query(
            serialize.GlycanCombination).join(
                serialize.GlycanCombination.components).join(
                    serialize.GlycanComposition.structure_classes).group_by(serialize.GlycanCombination.id)

        glycan_compositions = [
            c.convert() for c in glycan_query.all()]
        return glycan_compositions

    def _split_chromatograms_by_observation_priority(self, scored_chromatograms: List[TandemAnnotatedChromatogram],
                                                     minimum_ms1_score: float) -> Tuple[
                                                                                    GlycoformAggregator,
                                                                                    List[GlycopeptideChromatogramProxy]
                                                                                ]:
        proxies = [
            GlycopeptideChromatogramProxy.from_chromatogram(chrom)
            for chrom in scored_chromatograms
        ]

        by_structure: DefaultDict[Any, List[GlycopeptideChromatogramProxy]] = defaultdict(list)
        for proxy in proxies:
            by_structure[proxy.structure].append(proxy)

        best_instances: List[GlycopeptideChromatogramProxy] = []
        secondary_observations: List[GlycopeptideChromatogramProxy] = []
        for key, group in by_structure.items():
            group.sort(key=lambda x: x.total_signal, reverse=True)
            i = 0
            for i, member in enumerate(group):
                if member.source.score > minimum_ms1_score:
                    best_instances.append(member)
                    secondary_observations.extend(group[i + 1:])
                    break
                else:
                    secondary_observations.append(member)

        glycoform_agg = GlycoformAggregator(best_instances)
        return glycoform_agg, secondary_observations

    def _apply_revisions(self,
                         pipeline: GlycopeptideElutionTimeModelBuildingPipeline,
                         rt_model: GlycopeptideElutionTimeModelEnsemble,
                         revisions: List[GlycopeptideChromatogramProxy],
                         secondary_observations: List[GlycopeptideChromatogramProxy],
                         orphans: List[TandemSolutionsWithoutChromatogram],
                         updater: SpectrumMatchUpdater):
        was_updated: List[GlycopeptideChromatogramProxy] = []
        to_affirm: List[GlycopeptideChromatogramProxy] = []
        for rev in revisions:
            if rev.revised_from and rev.structure != rev.source.structure:
                was_updated.append(rev)
            else:
                to_affirm.append(rev)

        self.log("... Revising Secondary Occurrences")
        for rev in pipeline.revise_with(rt_model, secondary_observations):
            if rev.revised_from and rev.structure != rev.source.structure:
                was_updated.append(rev)
            else:
                to_affirm.append(rev)

        orphan_proxies = []
        for orphan in orphans:
            gpsm = orphan.best_match_for(orphan.structure)
            if not hasattr(gpsm.scan, "scan_time"):
                gpsm.scan = ScanInformation.from_scan(
                    updater.scan_loader.get_scan_by_id(
                        gpsm.scan.id))
            p = GlycopeptideChromatogramProxy.from_spectrum_match(
                gpsm, orphan)
            orphan_proxies.append(p)
        orphan_proxies = list(GlycoformAggregator(orphan_proxies).tag())

        self.log("... Revising Orphans")
        orphan_proxies: List[GlycopeptideChromatogramProxy] = pipeline.revise_with(
            rt_model, orphan_proxies)

        was_updated_orphans: List[GlycopeptideChromatogramProxy] = []
        to_affirm_orphans: List[GlycopeptideChromatogramProxy] = []
        for rev in orphan_proxies:
            if rev.revised_from and rev.structure != rev.source.structure:
                was_updated_orphans.append(rev)
            else:
                to_affirm_orphans.append(rev)

        self.log("... Updating best match assignments")

        revisions_accepted: List[RevisionSummary] = []
        # Use side-effects to update the source chromatogram
        for revision in was_updated:
            revisions_accepted.append(updater(revision)[1])

        # Use side-effects to update the source not-chromatogram
        for revised_orphan in was_updated_orphans:
            revisions.append(updater(revised_orphan)[1])

        self.log("... Affirming best match assignments with RT model")
        affirmed_results: List[RevisionSummary] = []
        for af in to_affirm:
            affirmed_results.append(updater.affirm_solution(af.source)[1])

        accepted_revisions = sum(rev.accepted for rev in revisions_accepted)
        affirmations_accepted = sum(rev.accepted for rev in affirmed_results)
        self.log(
            f"... Accepted {accepted_revisions}/{len(revisions_accepted)} "
            f"({accepted_revisions * 100.0 / (len(revisions_accepted) or 1):0.2f}%) revisions and "
            f"affirmed {affirmations_accepted}/{len(affirmed_results)} "
            f"({affirmations_accepted * 100.0 / (len(affirmed_results) or 1):0.2f}%) existing results")

        self.analysis_metadata['retention_time_revisions'] = {
            'accepted_revisions': accepted_revisions,
            'revisions_proposed': len(revisions_accepted),
            'accepted_affirmations': affirmations_accepted,
            'affirmations_proposed': len(affirmed_results)
        }


    def apply_retention_time_model(self, scored_chromatograms: Sequence[ChromatogramSolution],
                                   orphans: List[TandemSolutionsWithoutChromatogram],
                                   database, scan_loader: ProcessedRandomAccessScanSource,
                                   minimum_ms1_score: float=6.0) -> Tuple[
                                                                        List[ChromatogramSolution],
                                                                        List[TandemSolutionsWithoutChromatogram]
                                                                    ]:

        glycan_compositions = self._get_glycan_compositions_from_database(database)

        glycoform_agg, secondary_observations = self._split_chromatograms_by_observation_priority(
            scored_chromatograms, minimum_ms1_score=minimum_ms1_score)

        if not glycoform_agg.has_relative_pairs():
            self.log("... Insufficient identifications for Retention Time Modeling")
            return scored_chromatograms, orphans

        def threshold_fn(x):
            return x.q_value <= self.psm_fdr_threshold

        msn_match_args = self.make_msn_evaluation_kwargs()
        msn_match_args['error_tolerance'] = self.msn_mass_error_tolerance

        updater = SpectrumMatchUpdater(
            scan_loader,
            self.tandem_scoring_model,
            spectrum_match_cls=SpectrumMatch,
            id_maker=database.make_key_maker(),
            threshold_fn=threshold_fn,
            match_args=msn_match_args,
            fdr_estimator=self.fdr_estimator)

        revision_validator = CompoundRevisionValidator([
            PeptideYUtilizationPreservingRevisionValidator(
                spectrum_match_builder=updater),
            OxoniumIonRequiringUtilizationRevisionValidator(
                spectrum_match_builder=updater),
        ])

        self.log("... Begin Retention Time Modeling")

        pipeline = GlycopeptideElutionTimeModelBuildingPipeline(
            glycoform_agg, valid_glycans=glycan_compositions,
            revision_validator=revision_validator,
            n_workers=self.n_processes)

        rt_model, revisions = pipeline.run()
        if rt_model is None:
            self.retention_time_model = None
            return scored_chromatograms, orphans

        self.retention_time_model = rt_model
        rt_model.drop_chromatograms()
        updater.retention_time_model = rt_model

        self._apply_revisions(
            pipeline,
            rt_model,
            revisions,
            secondary_observations,
            orphans,
            updater)

        self.log("... Re-running chromatogram merge")
        merger = AnnotatedChromatogramAggregator(
            scored_chromatograms,
            require_unmodified=False,
            threshold_fn=threshold_fn)

        remerged_chromatograms = merger.run()

        out = []
        for chrom in remerged_chromatograms:
            if isinstance(chrom, ChromatogramSolution):
                chrom = chrom.chromatogram
            out.append(chrom)
        scored_chromatograms = self.score_chromatograms(out)

        return scored_chromatograms, orphans

    def _filter_out_poor_matches_before_saving(self,
                                               identified_glycopeptides: Sequence[IdentifiedGlycopeptide]) -> Sequence[
                                                                                                                IdentifiedGlycopeptide
                                                                                                            ]:
        filt = QValueRetentionStrategy(max(0.75, self.psm_fdr_threshold * 10.))
        for idgp in identified_glycopeptides:
            for gpsm_set in idgp.tandem_solutions:
                gpsm_set.select_top(filt)
        return identified_glycopeptides

    def _build_analysis_saved_parameters(self, identified_glycopeptides: Sequence[IdentifiedGlycopeptide],
                                         unassigned_chromatograms: ChromatogramFilter,
                                         chromatogram_extractor: ChromatogramExtractor,
                                         database):
        state = {
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
            "model_retention_time": self.model_retention_time,
            "retention_time_model": self.retention_time_model,
            "extra_evaluation_kwargs": self.extra_msn_evaluation_kwargs,
        }
        state['additional_metadata'] = self.analysis_metadata
        return state


    def _add_files_to_analysis(self):
        for path in self.file_manager.dir():
            if os.path.isdir(path):
                continue
            self.analysis.add_file(path, compress=True)
        self.log("Cleaning up temporary files")
        self.file_manager.clear()

    def save_solutions(self, identified_glycopeptides: List[IdentifiedGlycopeptide],
                       unassigned_chromatograms: ChromatogramFilter,
                       chromatogram_extractor: ChromatogramExtractor,
                       database):
        if self.analysis_name is None:
            return
        self.log("Saving Results To \"%s\"" % (self.database_connection,))
        analysis_saver = AnalysisSerializer(self.database_connection, self.sample_run_id, self.analysis_name)
        analysis_saver.set_peak_lookup_table(chromatogram_extractor.peak_mapping if chromatogram_extractor else {})
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
        self._add_files_to_analysis()
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
                 trust_precursor_fits=True, permute_decoy_glycans=False, rare_signatures=False,
                 model_retention_time=True, evaluation_kwargs=None):
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
            rare_signatures=rare_signatures,
            model_retention_time=model_retention_time,
            evaluation_kwargs=evaluation_kwargs)
        self.sample_path = sample_path
        self.output_path = output_path

    def make_peak_loader(self) -> ProcessedRandomAccessScanSource:
        peak_loader = ProcessedMSFileLoader(self.sample_path)
        if peak_loader.extended_index is None:
            if not peak_loader.has_index_file():
                self.log("Index file missing. Rebuilding.")
                peak_loader.build_extended_index()
            else:
                peak_loader.read_index_file()
            if peak_loader.extended_index is None or len(peak_loader.extended_index.msn_ids) < 1:
                raise ValueError("Sample Data Invalid: Could not validate MS/MS Index")
        return peak_loader

    def load_msms(self, peak_loader: ProcessedRandomAccessScanSource) -> List[ScanStub]:
        prec_info = peak_loader.precursor_information()
        msms_scans = [ScanStub(o, peak_loader) for o in prec_info if o.neutral_mass is not None]
        return msms_scans

    def _build_analysis_saved_parameters(self,
                                         identified_glycopeptides: List[IdentifiedGlycopeptide],
                                         unassigned_chromatograms: ChromatogramFilter,
                                         chromatogram_extractor: ChromatogramExtractor,
                                         database):
        state = {
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
            "search_strategy": GlycopeptideSearchStrategy.target_internal_decoy_competition.value,
            "trust_precursor_fits": self.trust_precursor_fits,
            "probing_range_for_missing_precursors": self.probing_range_for_missing_precursors,
            "scoring_model": self.peak_shape_scoring_model,
            "fdr_estimator": self.fdr_estimator,
            "permute_decoy_glycans": self.permute_decoy_glycans,
            "tandem_scoring_model": self.tandem_scoring_model,
            "rare_signatures": self.rare_signatures,
            "model_retention_time": self.model_retention_time,
            "retention_time_model": self.retention_time_model,
            "extra_evaluation_kwargs": self.extra_msn_evaluation_kwargs,
        }
        state['additional_metadata'] = self.analysis_metadata
        return state

    def make_analysis_serializer(self, output_path, analysis_name: str,
                                 sample_run,
                                 identified_glycopeptides: List[IdentifiedGlycopeptide],
                                 unassigned_chromatograms: ChromatogramFilter,
                                 database,
                                 chromatogram_extractor: ChromatogramExtractor):
        return GlycopeptideMSMSAnalysisSerializer(
            output_path, analysis_name, sample_run, identified_glycopeptides,
            unassigned_chromatograms, database, chromatogram_extractor)

    def save_solutions(self, identified_glycopeptides: List[IdentifiedGlycopeptide],
                       unassigned_chromatograms: ChromatogramFilter,
                       chromatogram_extractor: ChromatogramExtractor,
                       database):
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
        self._add_files_to_analysis()
        session = object_session(self.analysis)
        session.add(self.analysis)
        session.commit()
        self.log("Final Results Written To %s" % (self.output_path, ))


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
                 permute_decoy_glycans=False, rare_signatures=False,
                 model_retention_time=True, evaluation_kwargs=None):
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
            rare_signatures=rare_signatures, model_retention_time=model_retention_time,
            evaluation_kwargs=evaluation_kwargs)
        self.decoy_database_connection = decoy_database_connection
        self.use_decoy_correction_threshold = use_decoy_correction_threshold

    def make_search_engine(self, msms_scans: List[ScanStub],
                           database,
                           peak_loader: ProcessedRandomAccessScanSource):
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

    def _build_analysis_saved_parameters(self, identified_glycopeptides,
                                         unassigned_chromatograms,
                                         chromatogram_extractor,
                                         database):
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
            "search_strategy": GlycopeptideSearchStrategy.target_decoy_competition.value
        })
        return result

    def estimate_fdr(self, searcher: SearchEngineBase, target_decoy_set: TargetDecoySet):
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
                 glycan_score_threshold=1.0, mass_shifts=None,
                 n_processes=5, spectrum_batch_size=100,
                 maximum_mass=float('inf'), probing_range_for_missing_precursors=3,
                 trust_precursor_fits=True, use_memory_database=True,
                 fdr_estimation_strategy=None, glycosylation_site_models_path=None,
                 permute_decoy_glycans=False, fragile_fucose=False, rare_signatures=False,
                 extended_glycan_search=True, model_retention_time=True,
                 evaluation_kwargs=None,
                 oxonium_threshold=0.05,
                 peptide_masses_per_scan=60):
        if tandem_scoring_model == CoverageWeightedBinomialScorer:
            tandem_scoring_model = CoverageWeightedBinomialModelTree
        if fdr_estimation_strategy is None:
            fdr_estimation_strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
        super(MultipartGlycopeptideLCMSMSAnalyzer, self).__init__(
            database_connection, target_hypothesis_id, sample_path, output_path,
            analysis_name, grouping_error_tolerance, mass_error_tolerance,
            msn_mass_error_tolerance, psm_fdr_threshold, peak_shape_scoring_model,
            tandem_scoring_model, minimum_mass, save_unidentified,
            oxonium_threshold, None, mass_shifts,
            n_processes,
            spectrum_batch_size=spectrum_batch_size,
            use_peptide_mass_filter=True,
            maximum_mass=maximum_mass,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            trust_precursor_fits=trust_precursor_fits,
            # The multipart scoring algorithm automatically implies permute_decoy_glycan
            # fragment masses.
            permute_decoy_glycans=True,
            model_retention_time=model_retention_time,
            evaluation_kwargs=evaluation_kwargs)

        self.peptide_masses_per_scan = peptide_masses_per_scan
        self.fragile_fucose = fragile_fucose
        self.rare_signatures = rare_signatures
        self.extended_glycan_search = extended_glycan_search
        self.glycan_score_threshold = glycan_score_threshold
        self.decoy_database_connection = decoy_database_connection
        self.use_memory_database = use_memory_database
        self.decoy_hypothesis_id = decoy_hypothesis_id
        self.fdr_estimation_strategy = fdr_estimation_strategy
        self.glycosylation_site_models_path = glycosylation_site_models_path
        self.fdr_estimator = None
        self.retention_time_model = None
        self.precursor_mass_error_distribution = None
        self.localization_model = None

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

    def make_msn_evaluation_kwargs(self):
        kwargs = {
            "fragile_fucose": self.fragile_fucose,
            "rare_signatures": self.rare_signatures,
            "extended_glycan_search": self.extended_glycan_search,
        }
        kwargs.update(self.extra_msn_evaluation_kwargs)
        return kwargs

    def make_search_engine(self, msms_scans: List[ScanStub],
                           database,
                           peak_loader: ProcessedRandomAccessScanSource):
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
            cache_seeds=cache_seeds, evaluation_kwargs=self.make_msn_evaluation_kwargs(),
            oxonium_threshold=self.minimum_oxonium_ratio,
            peptide_masses_per_scan=self.peptide_masses_per_scan)
        return searcher

    def estimate_fdr(self, searcher: SearchEngineBase, target_decoy_set: SolutionSetGrouper):
        return searcher.estimate_fdr(target_decoy_set)

    def localize_modifications(self,
                               solution_sets: List[SpectrumSolutionSet],
                               scan_loader: ProcessedRandomAccessScanSource,
                               database: DiskBackedStructureDatabaseBase):

        hyp_parameters: Dict[str, Any] = database.hypothesis.parameters
        rules_by_name = {}
        table = None

        for rule in hyp_parameters.get('constant_modifications', []):
            if isinstance(rule, str):
                rule = rule_string_to_specialized_rule(rule)
            rules_by_name[rule.name] = rule
        for rule in hyp_parameters.get('variable_modifications', []):
            if table is not None:
                rule = table[rule]
            rules_by_name[rule.name] = rule

        self.log("Begin Modifications Localization")
        task = ScanLoadingModificationLocalizationSearcher(
            scan_loader,
            threshold_fn=lambda x: x.q_value <= self.psm_fdr_threshold,
            error_tolerance=self.msn_mass_error_tolerance,
            restricted_modifications=rules_by_name
        )

        solution_bins: List[EvaluatedSolutionBins] = []
        for i, sset in enumerate(solution_sets):
            if i % 1000 == 0 and i:
                self.log(f"... Processed {i} solution sets")
            solution_bin_set = task.process_solution_set(sset)
            solution_bins.append(solution_bin_set)

        training_examples = task.get_training_instances(solution_bins)
        self.log(f"{len(training_examples)} training examples for localization")

        ptm_prophet = task.train_ptm_prophet(training_examples)
        self.localization_model = task

        self.log("Scoring Site Localizations")
        task.select_top_isoforms(solution_bins, ptm_prophet)

    def apply_retention_time_model(self, scored_chromatograms: List[ChromatogramSolution],
                                   orphans: List[TandemSolutionsWithoutChromatogram],
                                   database,
                                   scan_loader: ProcessedRandomAccessScanSource,
                                   minimum_ms1_score=6.0):
        glycan_compositions = self._get_glycan_compositions_from_database(
            database)

        glycoform_agg, secondary_observations = self._split_chromatograms_by_observation_priority(
            scored_chromatograms, minimum_ms1_score=minimum_ms1_score)

        if not glycoform_agg.has_relative_pairs():
            self.log("... Insufficient identifications for Retention Time Modeling")
            return scored_chromatograms, orphans

        def threshold_fn(x):
            return x.q_value <= self.psm_fdr_threshold

        msn_match_args = self.make_msn_evaluation_kwargs()
        msn_match_args['error_tolerance'] = self.msn_mass_error_tolerance

        updater = SpectrumMatchUpdater(
            scan_loader,
            self.tandem_scoring_model,
            spectrum_match_cls=MultiScoreSpectrumMatch,
            id_maker=IdKeyMaker(glycan_compositions),
            threshold_fn=threshold_fn,
            match_args=msn_match_args,
            fdr_estimator=self.fdr_estimator)

        revision_validator = CompoundRevisionValidator([
            PeptideYUtilizationPreservingRevisionValidator(spectrum_match_builder=updater),
            OxoniumIonRequiringUtilizationRevisionValidator(spectrum_match_builder=updater),
        ])

        self.log("Begin Retention Time Modeling")

        pipeline = GlycopeptideElutionTimeModelBuildingPipeline(
            glycoform_agg, valid_glycans=glycan_compositions,
            revision_validator=revision_validator,
            n_workers=self.n_processes)

        rt_model, revisions = pipeline.run()
        if rt_model is None:
            self.retention_time_model = None
            return scored_chromatograms, orphans

        self.retention_time_model = rt_model
        rt_model.drop_chromatograms()
        updater.retention_time_model = rt_model

        self._apply_revisions(
            pipeline,
            rt_model,
            revisions,
            secondary_observations,
            orphans,
            updater)

        self.log("... Re-running chromatogram merge")
        merger = AnnotatedChromatogramAggregator(
            scored_chromatograms,
            require_unmodified=False,
            threshold_fn=threshold_fn)

        remerged_chromatograms = merger.run()

        out = []
        for chrom in remerged_chromatograms:
            if isinstance(chrom, ChromatogramSolution):
                chrom = chrom.chromatogram
            out.append(chrom)
        scored_chromatograms = self.score_chromatograms(out)

        return scored_chromatograms, orphans

    def finalize_solutions(self, identifications: List[identified_glycopeptide.IdentifiedGlycopeptide]):
        self.log("Back-filling localization information")
        k = 0
        for idgp in identifications:
            if not idgp.localizations:
                for sset in idgp.tandem_solutions:
                    k += 1
                    bins = self.localization_model.process_solution_set(sset)
                    self.localization_model.select_top_isoforms([bins])
        if k:
            self.log(f'Back-filled {k} spectra')

    def rank_target_hits(self, searcher: SearchEngineBase, target_decoy_set: TargetDecoySet):
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
        n_below_1 = 0
        n_below_5 = 0
        for target in target_hits:
            if target.q_value <= self.psm_fdr_threshold:
                n_below += 1
            if target.q_value <= 0.05:
                n_below_5 += 1
            if target.q_value <= 0.01:
                n_below_1 += 1
        self.log("%d spectrum matches accepted" % (n_below,))
        if self.psm_fdr_threshold != 0.05:
            self.log("%d spectra matched passing 5%% FDR" % n_below_5)
        if self.psm_fdr_threshold != 0.01:
            self.log("%d spectra matched passing 1%% FDR" % n_below_1)
        return target_hits

    def make_analysis_serializer(self, output_path, analysis_name, sample_run, identified_glycopeptides,
                                 unassigned_chromatograms, database, chromatogram_extractor):
        return DynamicGlycopeptideMSMSAnalysisSerializer(
            output_path, analysis_name, sample_run,
            identified_glycopeptides,
            unassigned_chromatograms, database, chromatogram_extractor)

    def map_chromatograms(self, searcher: SearchEngineBase,
                          extractor: ChromatogramExtractor,
                          target_hits: List[SpectrumSolutionSet],
                          revision_validator_type: Optional[Type[MS2RevisionValidator]] = None) -> Tuple[
                            ChromatogramFilter, TandemSolutionsWithoutChromatogram]:
        if revision_validator_type is None:
            revision_validator_type = SignalUtilizationMS2RevisionValidator
        return super().map_chromatograms(
            searcher,
            extractor,
            target_hits,
            revision_validator_type=revision_validator_type)

    def _build_analysis_saved_parameters(self, identified_glycopeptides, unassigned_chromatograms,
                                         chromatogram_extractor, database):
        database = GlycopeptideDiskBackedStructureDatabase(self.database_connection)
        result = super(MultipartGlycopeptideLCMSMSAnalyzer, self)._build_analysis_saved_parameters(
            identified_glycopeptides, unassigned_chromatograms,
            chromatogram_extractor, database)
        result.update({
            "target_database": str(self.database_connection),
            "decoy_database": str(self.decoy_database_connection),
            "search_strategy": GlycopeptideSearchStrategy.multipart_target_decoy_competition.value,
            "fdr_estimation_strategy": self.fdr_estimation_strategy,
            "fdr_estimator": self.fdr_estimator,
            "fragile_fucose": self.fragile_fucose,
            "rare_signatures": self.rare_signatures,
            "extended_glycan_search": self.extended_glycan_search,
            "glycosylation_site_models_path": self.glycosylation_site_models_path,
            "retention_time_model": self.retention_time_model,
            "localization_model": (self.localization_model.simplify()
                                   if self.localization_model is not None else None),
            "peptide_masses_per_scan": self.peptide_masses_per_scan,
        })
        return result
