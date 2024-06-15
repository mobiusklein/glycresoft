from typing import (Any, Callable, Collection, DefaultDict, Dict, Hashable,
                    List, Optional, Set, Tuple, NamedTuple, Type, Union,
                    TYPE_CHECKING)

from glycresoft.chromatogram_tree import Chromatogram, ChromatogramFilter
from glycresoft.chromatogram_tree.mass_shift import Unmodified

from glycresoft.task import TaskBase

from .base import Predicate, SolutionEntry, TargetType, default_threshold, DEBUG_MODE
from .chromatogram import TandemAnnotatedChromatogram
from .revision import MS2RevisionValidator
from .graph import build_glycopeptide_key, MassShiftDeconvolutionGraph, RepresenterDeconvolution


class GraphAnnotatedChromatogramAggregator(TaskBase):
    annotated_chromatograms: List[TandemAnnotatedChromatogram]
    delta_rt: float
    require_unmodified: bool
    threshold_fn: Predicate
    key_fn: Callable[[TargetType], Hashable]

    revision_validator: MS2RevisionValidator

    def __init__(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                 threshold_fn=default_threshold, key_fn=build_glycopeptide_key,
                 revision_validator: Optional[MS2RevisionValidator] = None):
        if revision_validator is None:
            revision_validator = MS2RevisionValidator(threshold_fn)
        self.annotated_chromatograms = sorted(annotated_chromatograms,
                                              key=lambda x: (
                                                  max(sset.best_solution(
                                                  ).score for sset in x.tandem_solutions)
                                                  if x.tandem_solutions else -float('inf'), x.total_signal),
                                              reverse=True)
        self.delta_rt = delta_rt
        self.require_unmodified = require_unmodified
        self.threshold_fn = threshold_fn
        self.key_fn = key_fn
        self.revision_validator = revision_validator

    def build_graph(self) -> MassShiftDeconvolutionGraph:
        self.log("Constructing chromatogram graph")
        assignable = []
        rest: List[Chromatogram] = []
        for chrom in self.annotated_chromatograms:
            if chrom.composition is not None:
                assignable.append(chrom)
            else:
                rest.append(chrom)
        graph = MassShiftDeconvolutionGraph(assignable)
        graph.build(self.delta_rt, self.threshold_fn)
        return graph, rest

    def deconvolve(self, graph: MassShiftDeconvolutionGraph):
        components = graph.connected_components()
        assigned = []
        if DEBUG_MODE:
            breakpoint()

        for group in components:
            spectra_in = set()
            for node in group:
                for sset in node.tandem_solutions:
                    spectra_in.add(sset.scan.id)

            deconv = RepresenterDeconvolution(
                group,
                threshold_fn=self.threshold_fn,
                key_fn=self.key_fn,
                revision_validator=self.revision_validator,
            )
            deconv.solve()
            reps = deconv.assign_representers()
            spectra_out = set()
            for node in reps:
                for sset in node.tandem_solutions:
                    spectra_out.add(sset.scan.id)
            if spectra_out != spectra_in:
                self.error(
                    f"{len(spectra_in - spectra_out)} spectra lost while solving this component")
            if DEBUG_MODE and spectra_out != spectra_in:
                breakpoint()
            assigned.extend(reps)
        return assigned

    def run(self):
        graph, rest = self.build_graph()
        solutions = self.deconvolve(graph)
        solutions.extend(rest)
        return ChromatogramFilter(solutions)


class AnnotatedChromatogramAggregator(TaskBase):
    """A basic chromatogram merger that groups chromatograms according to their best analyte.

    This algorithm looks for RT proximity.
    """

    annotated_chromatograms: ChromatogramFilter
    delta_rt: float
    require_unmodified: bool
    threshold_fn: Predicate

    def __init__(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                 threshold_fn=default_threshold):
        self.annotated_chromatograms = annotated_chromatograms
        self.delta_rt = delta_rt
        self.require_unmodified = require_unmodified
        self.threshold_fn = threshold_fn

    def combine_chromatograms(self, aggregated):
        merged = []
        for entity, group in aggregated.items():
            out = []
            group = sorted(group, key=lambda x: x.start_time)
            chroma = group[0]
            for obs in group[1:]:
                if chroma.chromatogram.overlaps_in_time(obs) or abs(
                        chroma.end_time - obs.start_time) < self.delta_rt:
                    chroma = chroma.merge(obs)
                else:
                    out.append(chroma)
                    chroma = obs
            out.append(chroma)
            merged.extend(out)
        return merged

    def aggregate_by_annotation(self, annotated_chromatograms: ChromatogramFilter) -> Tuple[
                                                                                            List[TandemAnnotatedChromatogram],
                                                                                            DefaultDict[
                                                                                                str,
                                                                                                List[TandemAnnotatedChromatogram]
                                                                                            ]
                                                                                        ]:
        """Group chromatograms by their assigned entity."""
        finished = []
        aggregated = DefaultDict(list)
        for chroma in annotated_chromatograms:
            if chroma.composition is not None:
                if chroma.entity is not None:
                    # Convert to string to avoid redundant sequences from getting
                    # binned differently due to random ordering of ids.
                    aggregated[str(chroma.entity)].append(chroma)
                else:
                    aggregated[str(chroma.composition)].append(chroma)
            else:
                finished.append(chroma)
        return finished, aggregated

    def aggregate(self, annotated_chromatograms: ChromatogramFilter) -> List[TandemAnnotatedChromatogram]:
        """Drives the aggregation process."""
        self.log("Aggregating Common Entities: %d chromatograms" %
                 (len(annotated_chromatograms,)))
        finished, aggregated = self.aggregate_by_annotation(
            annotated_chromatograms)
        finished.extend(self.combine_chromatograms(aggregated))
        self.log("After Merging: %d chromatograms" % (len(finished),))
        return finished

    def replace_solution(self, chromatogram: TandemAnnotatedChromatogram, solutions: List[SolutionEntry]):
        # select the best solution
        solutions = sorted(
            solutions, key=lambda x: x.score, reverse=True)

        # remove the invalidated mass shifts so that we're dealing with the raw masses again.
        current_shifts = chromatogram.chromatogram.mass_shifts
        if len(current_shifts) > 1:
            self.log("... Found a multiply shifted identification with no Unmodified state: %s" % (
                chromatogram.entity, ))
        partitions = []
        for shift in current_shifts:
            # The _ collects the "not modified with shift" portion of the chromatogram
            # we'll strip out in successive iterations. By virtue of reaching this point,
            # we're never dealing with an Unmodified portion.
            partition, _ = chromatogram.chromatogram.bisect_mass_shift(shift)
            # If we're somehow dealing with a compound mass shift here, remove
            # the bad shift from the composite, and reset the modified nodes to
            # Unmodified.
            partitions.append(partition.deduct_node_type(shift))

        # Merge in
        accumulated_chromatogram = partitions[0]
        for partition in partitions[1:]:
            accumulated_chromatogram = accumulated_chromatogram.merge(
                partition)
        chromatogram.chromatogram = accumulated_chromatogram

        # update the tandem annotations
        for solution_set in chromatogram.tandem_solutions:
            solution_set.mark_top_solutions(reject_shifted=self.require_unmodified)
        chromatogram.assign_entity(
            solutions[0],
            entity_chromatogram_type=chromatogram.chromatogram.__class__)
        chromatogram.representative_solutions = solutions
        return chromatogram

    def reassign_modified_only_cases(self, annotated_chromatograms):
        out = []
        for chromatogram in annotated_chromatograms:
            # the structure's best match has not been identified in an unmodified state
            if Unmodified not in chromatogram.mass_shifts:
                original_entity = getattr(chromatogram, "entity")
                solutions = chromatogram.most_representative_solutions(
                    self.threshold_fn, reject_shifted=True)
                # if there is a reasonable solution in an unmodified state
                if solutions:
                    chromatogram = self.replace_solution(
                        chromatogram, solutions)
                    self.debug("... Replacing %s with %s",
                               original_entity, chromatogram.entity)
                    out.append(chromatogram)
                else:
                    self.log(
                        "... Could not find an alternative option for %r" % (chromatogram,))
                    out.append(chromatogram)
            else:
                out.append(chromatogram)
        return out

    def run(self):
        merged_chromatograms = self.aggregate(self.annotated_chromatograms)
        if self.require_unmodified:
            spliced = self.reassign_modified_only_cases(merged_chromatograms)
            merged_chromatograms = self.aggregate(spliced)
        result = ChromatogramFilter(merged_chromatograms)
        return result


def aggregate_by_assigned_entity(annotated_chromatograms: List[TandemAnnotatedChromatogram],
                                 delta_rt: float = 0.25,
                                 require_unmodified: bool = True,
                                 threshold_fn: Predicate = default_threshold,
                                 revision_validator: Optional[MS2RevisionValidator] = None):
    job = GraphAnnotatedChromatogramAggregator(
        annotated_chromatograms,
        delta_rt=delta_rt,
        require_unmodified=require_unmodified,
        threshold_fn=threshold_fn,
        revision_validator=revision_validator
    )
    finished = job.run()
    return finished
