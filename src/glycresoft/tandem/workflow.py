from typing import Optional, Sequence, Iterable, TypeVar

from glycresoft.task import TaskBase

from .spectrum_match.solution_set import SpectrumSolutionSet
from .target_decoy import TargetDecoySet


T = TypeVar('T')


def chunkiter(collection: Sequence[T], size: int=200) -> Iterable[Sequence[T]]:
    i = 0
    while collection[i:(i + size)]:
        yield collection[i:(i + size)]
        i += size


def format_identification(spectrum_solution: SpectrumSolutionSet) -> str:
    return "%s:%0.3f:(%0.3f) ->\n%s" % (
        spectrum_solution.scan.id,
        spectrum_solution.scan.precursor_information.neutral_mass,
        spectrum_solution.best_solution().score,
        str(spectrum_solution.best_solution().target))


def format_identification_batch(group: Iterable[SpectrumSolutionSet], n: int) -> str:
    representers = dict()
    group = sorted(group, key=lambda x: x.score, reverse=True)
    for ident in group:
        key = str(ident.best_solution().target)
        if key in representers:
            continue
        else:
            representers[key] = ident
    to_represent = sorted(
        representers.values(), key=lambda x: x.score, reverse=True)
    return '\n'.join(map(format_identification, to_represent[:n]))


class SearchEngineBase(TaskBase):

    def search(self, precursor_error_tolerance: float=1e-5,
               simplify: bool=True,
               batch_size: int=500,
               limit: Optional[int]=None, *args, **kwargs) -> TargetDecoySet:
        raise NotImplementedError()

    def estimate_fdr(self, *args, **kwargs) -> TargetDecoySet:
        raise NotImplementedError()
