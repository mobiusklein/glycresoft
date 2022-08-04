from typing import List, Tuple, Callable, NamedTuple, Union, Dict, Optional, Counter

from ms_deisotope.data_source import ProcessedScan
from ms_deisotope.output import ProcessedMzMLLoader

from glycopeptidepy.structure import PeptideSequence, Modification, ModificationRule

from glycan_profiling.task import TaskBase
from glycan_profiling.structure import FragmentCachingGlycopeptide

from glycan_profiling.tandem.peptide.scoring import localize

from glycan_profiling.tandem.spectrum_match import SpectrumMatch, SpectrumSolutionSet, LocalizationScore

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


class ModificationLocalizationSearcher(TaskBase):
    scan_loader: ProcessedMzMLLoader
    threshold_fn: Callable[[SpectrumMatch], bool]
    restricted_modifications: Dict[str, ModificationRule]

    def __init__(self, scan_loader, threshold_fn=lambda x: x.q_value < 0.05,
                 error_tolerance: float = 2e-5,
                 restricted_modifications: Optional[Dict[str, ModificationRule]] = None):
        if restricted_modifications is None:
            restricted_modifications = {}
        self.scan_loader = scan_loader
        self.threshold_fn = threshold_fn
        self.error_tolerance = error_tolerance
        self.restricted_modifications = restricted_modifications

    def get_modifications_for_peptide(self, peptide: FragmentCachingGlycopeptide) -> Tuple[Tuple[Modification],
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

    def resolve_spectrum(self, scan_ref) -> ProcessedScan:
        return self.scan_loader.get_scan_by_id(scan_ref.scan_id)

    def find_overlapping_localization_solutions(self, solution_set: SpectrumSolutionSet) -> List[Tuple[List[SpectrumMatch],
                                                                                                       PeptideGroupToken]]:
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

    def process_solution_set(self, solution_set: SpectrumSolutionSet) -> List[Tuple[List[SpectrumMatch], List[localize.PTMProphetEvaluator]]]:
        solution_bins = self.find_overlapping_localization_solutions(
            solution_set)
        scan = self.resolve_spectrum(solution_set)
        solutions = [(solution_bin, self.process_localization_bin(scan, signature))
                     for solution_bin, signature in solution_bins]
        return solutions

    def process_localization_bin(self,
                                 scan: ProcessedScan,
                                 modification_group_token: PeptideGroupToken):
        solutions = []
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
            solutions.append(el)
        return solutions

