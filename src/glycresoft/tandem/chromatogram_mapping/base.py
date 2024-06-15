import os
import logging
from typing import (Any, Callable, Hashable,
                    List, Tuple, NamedTuple, Union)

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase

from glycresoft.structure.structure_loader import FragmentCachingGlycopeptide

from ..spectrum_match.spectrum_match import SpectrumMatch
from ..spectrum_match.solution_set import NOParsimonyMixin


DEBUG_MODE = bool(os.environ.get('GLYCRESOFTCHROMATOGRAMDEBUG', False))

logger = logging.getLogger("glycresoft.chromatogram_mapping")
logger.addHandler(logging.NullHandler())


def always(x):
    return True


def default_threshold(x):
    return x.q_value < 0.05


Predicate = Callable[[Any], bool]
TargetType = Union[Any, FragmentCachingGlycopeptide]


class SolutionEntry(NamedTuple):
    solution: Any
    score: float
    percentile: float
    best_score: float
    match: SpectrumMatch


class NOParsimonyRepresentativeSelector(NOParsimonyMixin):
    def get_score(self, solution: SolutionEntry) -> float:
        return solution.percentile

    def get_target(self, solution: SolutionEntry) -> Any:
        return solution.match.target

    def sort(self, solution_set: List[SolutionEntry]) -> List[SolutionEntry]:
        solution_set = sorted(
            solution_set, key=lambda x: x.percentile, reverse=True)
        try:
            if solution_set and self.get_target(solution_set[0]).is_o_glycosylated():
                solution_set = self.hoist_equivalent_n_linked_solution(
                    solution_set)
        except AttributeError:
            import warnings
            warnings.warn("Could not determine glycosylation state of target of type %r" % type(
                self.get_target(solution_set[0])))
        return solution_set

    def __call__(self, solution_set: List[SolutionEntry]) -> List[SolutionEntry]:
        return self.sort(solution_set)


parsimony_sort = NOParsimonyRepresentativeSelector()


def build_glycopeptide_key(solution) -> Tuple:
    structure = solution
    key = (str(structure.clone().deglycosylate()),
           str(structure.glycan_composition))
    return key


KeyTargetMassShift = Tuple[Hashable, TargetType, MassShiftBase]
KeyMassShift = Tuple[Hashable, MassShiftBase]
