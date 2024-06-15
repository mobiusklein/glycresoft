import math

from array import array
from typing import Any, DefaultDict, List, Tuple, Callable, NamedTuple, Union, Dict, Optional, Counter
from glycresoft.structure.scan import ScanStub, ScanWrapperBase, ScanInformation
from glycresoft.tandem.ref import SpectrumReference

from ms_deisotope.data_source import ProcessedScan
from ms_deisotope.output import ProcessedMzMLLoader

from glycopeptidepy.structure import PeptideSequence, Modification, ModificationRule

from glycresoft.task import TaskBase
from glycresoft.structure import FragmentCachingGlycopeptide
from glycresoft.structure.probability import KDModel, MultiKDModel

from glycresoft.tandem.peptide.scoring import localize

from glycresoft.tandem.spectrum_match import SpectrumMatch, SpectrumSolutionSet, LocalizationScore

from glypy.structure.glycan_composition import HashableGlycanComposition


class PeptideGroupToken(NamedTuple):
    peptide: PeptideSequence
    modifications: Tuple[Tuple[Union[Modification, ModificationRule], int]]
    glycan_composition: HashableGlycanComposition

    def __eq__(self, other: 'PeptideGroupToken'):
        if not self.peptide.base_sequence_equality(other.peptide):
            return False
        if self.glycan_composition != other.glycan_composition:
            return False
        if self.modifications != other.modifications:
            return False
        return True

    def __ne__(self, other: 'PeptideGroupToken'):
        return not self == other

    def __hash__(self):
        code = hash(self.glycan_composition) & hash(self.modifications)
        code &= len(self.peptide)
        code &= hash(self.peptide[0].symbol)
        return code


SolutionsBin = Tuple[List[SpectrumMatch], PeptideGroupToken]
EvaluatedSolutionBins = Tuple[List[SpectrumMatch],
                              List[localize.PTMProphetEvaluator]]

ScanOrRef = Union[ProcessedScan, ScanStub,
                  SpectrumReference, ScanWrapperBase, ScanInformation]


class LocalizationGroup(NamedTuple):
    spectrum_matches: List[SpectrumMatch]
    localization_matches: List[localize.PTMProphetEvaluator]


class EvaluatedSolutionBins(NamedTuple):
    solution_set: SpectrumSolutionSet
    groups: List[LocalizationGroup]

    @property
    def scan_id(self):
        return self.solution_set.scan.scan_id


class ModificationLocalizationSearcher(TaskBase):
    error_tolerance: float
    threshold_fn: Callable[[SpectrumMatch], bool]
    restricted_modifications: Dict[str, ModificationRule]
    model: Optional[MultiKDModel]

    def __init__(self,
                 threshold_fn: Callable[[SpectrumMatch],
                                        bool] = lambda x: x.q_value < 0.05,
                 error_tolerance: float = 2e-5,
                 restricted_modifications: Optional[Dict[str, ModificationRule]] = None,
                 model: Optional[MultiKDModel]=None):
        if restricted_modifications is None:
            restricted_modifications = {}
        self.threshold_fn = threshold_fn
        self.error_tolerance = error_tolerance
        self.restricted_modifications = restricted_modifications
        self.model = model

    def get_modifications_for_peptide(self, peptide: FragmentCachingGlycopeptide) -> Tuple[Tuple[Tuple[Modification, int]],
                                                                                           HashableGlycanComposition]:
        modifications = Counter()
        for position in peptide:
            if position.modifications:
                mod = position.modifications[0].rule
                if mod.name in self.restricted_modifications:
                    mod = self.restricted_modifications[mod.name]
                modifications[mod] += 1
        modifications = list(modifications.items())
        modifications.sort(key=lambda x: x[0].name)
        glycan = None
        if peptide.glycan_composition:
            glycan = peptide.glycan_composition
        return tuple(modifications), glycan

    def find_overlapping_localization_solutions(self, solution_set: SpectrumSolutionSet) -> List[Tuple[List[SpectrumMatch],
                                                                                                       PeptideGroupToken]]:
        if isinstance(solution_set, SpectrumMatch):
            solution_set = [solution_set]
        bins: List[Tuple[List[SpectrumMatch],
                         PeptideGroupToken]] = []
        for sm in solution_set:
            if not self.threshold_fn(sm):
                continue
            target: FragmentCachingGlycopeptide = sm.target
            mod_signature, glycan = self.get_modifications_for_peptide(target)
            token = PeptideGroupToken(target, mod_signature, glycan)
            for solution_bin, bin_token in bins:
                if bin_token == token:
                    solution_bin.append(sm)
                    break
            else:
                bins.append(([sm], token))
        return bins

    def process_localization_bin(self,
                                 scan: ProcessedScan,
                                 modification_group_token: PeptideGroupToken) -> List[localize.PTMProphetEvaluator]:
        solutions = []
        scan_description = ScanInformation.from_scan(scan)
        for mod_sig, count in modification_group_token.modifications:
            el = localize.PTMProphetEvaluator(
                scan,
                modification_group_token.peptide,
                modification_rule=mod_sig,
                modification_count=count
            )
            el.score_arrangements(
                error_tolerance=self.error_tolerance
            )
            # At this point we don't need the full scan representation anymore,
            # so we can drop the peaks later.
            el.scan = scan_description
            solutions.append(el)
        return solutions

    def get_training_instances(self, solutions: List[EvaluatedSolutionBins]) -> List[localize.PTMProphetEvaluator]:
        candidates = []
        for sset_bin in solutions:
            for sol in sset_bin.groups:
                for loc in sol.localization_matches:
                    if len(loc.peptidoforms) > 1:
                        candidates.append(loc)
        return candidates

    def train_ptm_prophet(self, training_instances: List[localize.PTMProphetEvaluator],
                          maxiter: int = 150) -> MultiKDModel:
        o_scores = array('d')
        m_scores = array('d')
        for inst in training_instances:
            for scores in inst.solution_for_site.values():
                if not math.isfinite(scores.o_score) or not math.isfinite(scores.m_score):
                    continue
                o_scores.append(scores.o_score)
                m_scores.append(scores.m_score)

        while len(o_scores) < 2:
            o_scores.append(0.99)
            m_scores.append(0.99)
        o_scores.append(0.5)
        m_scores.append(0.5)

        prophet = MultiKDModel(0.5, [KDModel(), KDModel()])
        for o, m in zip(o_scores, m_scores):
            prophet.add(o, [o, m])

        self.log("Begin fitting localization score model")
        try:
            it, delta = prophet.fit(maxiter)
        except ValueError as err:
            self.log(f"Failed to fit localization model: {err}")
            self.model = None
            return None
        if it < maxiter:
            self.log(
                f"Localization score model converged after {it} iterations")
        else:
            self.log(
                f"The localization model failed to converge after {it} iterations ({delta})")
        self.model = prophet
        if prophet is not None:
            prophet.close_thread_pool()
        return prophet

    def _select_top_isoform_in_bin(self, sol: LocalizationGroup, prophet: Optional[MultiKDModel]=None):
        isoform_scores: DefaultDict[str,
                                    List[LocalizationScore]] = DefaultDict(list)
        isoform_weights: DefaultDict[str, float] = DefaultDict(float)
        for loc in sol.localization_matches:
            seen = set()
            for iso, scores, weight in loc.score_isoforms(prophet=prophet):
                key = str(iso.peptide)
                if key in seen:
                    continue
                isoform_scores[key].extend(scores)
                isoform_weights[key] += weight

        max_weight = max(isoform_weights.values())
        for psm in sol.spectrum_matches:
            key = str(psm.target)
            if key in isoform_scores:
                psm.localizations = isoform_scores[key]
            if abs(isoform_weights[key] - max_weight) > 1e-2 and psm.best_match:
                psm.best_match = False
                psm.valid = False
                self.log(f"... Invalidating {psm.target}@{psm.scan_id}")

    def select_top_isoforms(self, solutions: List[List[EvaluatedSolutionBins]], prophet: Optional[MultiKDModel]=None):
        if prophet is None:
            prophet = self.model
        for sset_bin in solutions:
            for sol in sset_bin.groups:
                self._select_top_isoform_in_bin(sol, prophet=prophet)

    def process_solution_set(self, solution_set: SpectrumSolutionSet) -> EvaluatedSolutionBins:
        solution_bins = self.find_overlapping_localization_solutions(
            solution_set)
        scan = self.resolve_spectrum(solution_set)
        solutions = [LocalizationGroup(solution_bin, self.process_localization_bin(scan, signature))
                     for solution_bin, signature in solution_bins]
        return EvaluatedSolutionBins(solution_set, solutions)

    def resolve_spectrum(self, scan_ref: ScanOrRef) -> ProcessedScan:
        if isinstance(scan_ref, ScanWrapperBase):
            scan_ref = scan_ref.scan
        if isinstance(scan_ref, ProcessedScan):
            return scan_ref
        raise TypeError(type(scan_ref))

    def with_scan_loader(self, scan_loader):
        dup = ScanLoadingModificationLocalizationSearcher(
            scan_loader, self.threshold_fn, self.error_tolerance, self.restricted_modifications)
        dup.model = self.model
        return dup



class ScanLoadingModificationLocalizationSearcher(ModificationLocalizationSearcher):
    scan_loader: ProcessedMzMLLoader

    def __init__(self, scan_loader: ProcessedMzMLLoader,
                 threshold_fn: Callable[[SpectrumMatch], bool]=lambda x: x.q_value < 0.05,
                 error_tolerance: float = 2e-5,
                 restricted_modifications: Optional[Dict[str, ModificationRule]] = None,
                 model: Optional[MultiKDModel]=None):
        self.scan_loader = scan_loader
        super().__init__(threshold_fn, error_tolerance, restricted_modifications, model)

    def __reduce__(self):
        return self.__class__, (None, self.threshold_fn, self.error_tolerance, self.restricted_modifications)

    def resolve_spectrum(self, scan_ref: ScanOrRef) -> ProcessedScan:
        if isinstance(scan_ref, ScanWrapperBase):
            scan_ref = scan_ref.scan
        if isinstance(scan_ref, ProcessedScan):
            return scan_ref
        if self.scan_loader is None:
            raise TypeError("Cannot load spectrum by reference when the `scan_loader` attribute is not set")
        return self.scan_loader.get_scan_by_id(scan_ref.scan_id)

    def simplify(self):
        return ModificationLocalizationSearcher(
            self.threshold_fn,
            self.error_tolerance,
            self.restricted_modifications,
            self.model)
