from typing import Any, Callable, Dict, List, Tuple, Type, NamedTuple

from ms_deisotope.data_source import ProcessedRandomAccessScanSource, ProcessedScan

from glycan_profiling.task import TaskBase

from glycan_profiling.chromatogram_tree import Chromatogram, ChromatogramFilter, MassShift, mass_shift


from glycan_profiling.tandem.target_decoy import FDREstimatorBase

from glycan_profiling.tandem.glycopeptide.identified_structure import IdentifiedGlycopeptide

from glycan_profiling.tandem.spectrum_match.spectrum_match import MultiScoreSpectrumMatch, SpectrumMatch, SpectrumMatcherBase
from glycan_profiling.tandem.spectrum_match import SpectrumSolutionSet, MultiScoreSpectrumSolutionSet


class CoElutionCandidate(NamedTuple):
    reference: IdentifiedGlycopeptide
    extensions: List[Chromatogram]
    ms2_spectra: List[MultiScoreSpectrumMatch]
    mass_shift: MassShift

    def merge(self, scan_id_to_solution_set: Dict[str, SpectrumSolutionSet]) -> Tuple[IdentifiedGlycopeptide, List[IdentifiedGlycopeptide]]:
        ssets = []
        to_drop = []
        for gpsm in self.ms2_spectra:
            if gpsm.scan_id in scan_id_to_solution_set:
                sset = scan_id_to_solution_set[gpsm.scan_id]
                sset.append(gpsm)
                ssets.append(sset)
            else:
                ssets.append(MultiScoreSpectrumSolutionSet(gpsm.scan, [gpsm]))

        for part in self.extensions:
            if not isinstance(part, IdentifiedGlycopeptide):
                self.reference.chromatogram.merge_in_place(part, node_type=self.mass_shift)
            else:
                to_drop.append(part)
                self.reference.chromatogram.merge_in_place(
                    part.chromatogram, node_type=self.mass_shift)

        return self.reference, to_drop


class CoElutionAdductFinder(TaskBase):
    scan_loader: ProcessedRandomAccessScanSource
    chromatograms: ChromatogramFilter

    msn_scoring_model: Type[SpectrumMatcherBase]
    msn_evaluation_args: Dict[str, Any]
    fdr_estimator: FDREstimatorBase

    scan_id_to_solution_set: Dict[str, SpectrumSolutionSet]
    threshold_fn: Callable[[SpectrumMatch], bool]

    def __init__(self, scan_loader: ProcessedRandomAccessScanSource,
                 chromatograms: ChromatogramFilter,
                 msn_scoring_model: SpectrumMatcherBase,
                 msn_evaluation_args: Dict[str, Any],
                 fdr_estimator: FDREstimatorBase,
                 scan_id_to_solution_set: Dict[str, SpectrumSolutionSet],
                 threshold_fn: Callable[[SpectrumMatch], bool]=lambda x: x.q_value < 0.05):
        self.scan_loader = scan_loader
        self.chromatograms = chromatograms

        self.msn_scoring_model = msn_scoring_model
        self.msn_evaluation_args = msn_evaluation_args

        self.fdr_estimator = fdr_estimator
        self.threshold_fn = threshold_fn

        self.scan_id_to_solution_set = scan_id_to_solution_set

    def find_coeluting(
        self,
        reference: IdentifiedGlycopeptide,
        mass: float,
        error_tolerance: float = 2e-5,
    ) -> List[Chromatogram]:
        alternates = [
            alt for alt in self.chromatograms.find_all_by_mass(mass, error_tolerance)
            if reference.overlaps_in_time(alt)
        ]
        return alternates

    def find_ms2_spectra_for_chromatogram(self, chromatogram: Chromatogram,
                                          error_tolerance: float) -> List[ProcessedScan]:
        pinfos = self.scan_loader.msms_for(
            chromatogram.weighted_neutral_mass,
            error_tolerance,
            start_time=chromatogram.start_time - 3,
            end_time=chromatogram.end_time + 3,
        )
        return [p.product for p in pinfos]

    def evaluate_msn_spectra(self, reference: IdentifiedGlycopeptide,
                             adduct: MassShift,
                             product_spectra: List[ProcessedScan]) -> List[MultiScoreSpectrumMatch]:
        acc = []
        expects_tandem_shift = adduct.tandem_mass != 0
        for product in product_spectra:
            match: SpectrumMatcherBase = self.msn_scoring_model.evaluate(
                product,
                reference.structure,
                mass_shift=adduct,
                **self.msn_evaluation_args
            )

            modified_peaks = []
            for pfp in match.solution_map:
                chem_shift = pfp.fragment.chemical_shift
                if chem_shift and chem_shift.name == adduct.name:
                    modified_peaks.append(pfp)

            if expects_tandem_shift and not modified_peaks:
                continue

            match: MultiScoreSpectrumMatch = MultiScoreSpectrumMatch.from_match_solution(match)

            self.fdr_estimator.score_all([match])
            if product.scan_id in self.scan_id_to_solution_set:
                sset = self.scan_id_to_solution_set[product.scan_id]
                alt_match = sset.best_solution()
                if alt_match.score > match.score:
                    continue

            if self.threshold_fn(match):
                acc.append(match)
        return acc

    def coeluting_adduction_candidates(
        self,
        identified_structures: List[IdentifiedGlycopeptide],
        adduct: MassShift,
        error_tolerance: float = 1e-5,
    ) -> List[CoElutionCandidate]:
        results = []
        n = len(identified_structures)
        for i, ref in enumerate(identified_structures):
            if i % 250 == 0:
                self.log(f"{i}/{n} ({i / n * 100.0:0.2f}%) {len(results)} cases found")
            alts = self.find_coeluting(
                ref,
                ref.weighted_neutral_mass + adduct.mass,
                error_tolerance
            )
            tandem_candidates: List[ProcessedScan] = []
            for alt in alts:
                tandem_candidates.extend(
                    self.find_ms2_spectra_for_chromatogram(alt, error_tolerance))
            acc = self.evaluate_msn_spectra(ref, adduct, tandem_candidates)
            if acc:
                results.append(
                    CoElutionCandidate(ref, alts, acc, adduct))
        return results
