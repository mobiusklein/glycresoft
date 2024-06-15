from typing import (Any, Callable, Collection, DefaultDict, Dict, Hashable,
                    List, Optional, Set, Tuple, NamedTuple, Type, Union,
                    TYPE_CHECKING)

from glycresoft.chromatogram_tree.mass_shift import MassShiftBase, Unmodified
from glycresoft.chromatogram_tree.relation_graph import (
    ChromatogramGraph, ChromatogramGraphEdge, ChromatogramGraphNode, TimeQuery)

from glycresoft.task import TaskBase

from ..spectrum_match import SpectrumMatch

from .base import Predicate, always, SolutionEntry, TargetType, parsimony_sort, KeyMassShift, KeyTargetMassShift, build_glycopeptide_key
from .revision import MS2RevisionValidator, SpectrumMatchBackFiller


if TYPE_CHECKING:
    from .chromatogram import TandemAnnotatedChromatogram


class MassShiftDeconvolutionGraphNode(ChromatogramGraphNode):
    chromatogram: 'TandemAnnotatedChromatogram'

    def __init__(self, chromatogram, index, edges=None):
        super(MassShiftDeconvolutionGraphNode, self).__init__(
            chromatogram, index, edges=edges)

    def most_representative_solutions(self, threshold_fn: Predicate = always, reject_shifted: bool = False, percentile_threshold: float = 1e-5) -> List['SolutionEntry']:
        """Find the most representative solutions, the (very nearly the same, hopefully) structures with
        the highest aggregated score across all MSn events assigned to this collection.

        Parameters
        ----------
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False
        percentile_threshold : float, optional
            The difference between the worst and best percentile to be reported. Defaults to 1e-5.

        Returns
        -------
        list
            A list of solutions with approximately the greatest weight
        """
        return self.chromatogram.most_representative_solutions(
            threshold_fn=threshold_fn, reject_shifted=reject_shifted,
            percentile_threshold=percentile_threshold
        )

    def solutions_for(self, structure, threshold_fn: Predicate = always, reject_shifted: bool = False) -> List[SpectrumMatch]:
        return self.chromatogram.solutions_for(
            structure, threshold_fn=threshold_fn, reject_shifted=reject_shifted)

    @property
    def mass_shifts(self):
        return self.chromatogram.mass_shifts

    @property
    def tandem_solutions(self):
        return self.chromatogram.tandem_solutions

    @property
    def total_signal(self):
        return self.chromatogram.total_signal

    @property
    def entity(self):
        return self.chromatogram.entity

    def best_match_for(self, structure, threshold_fn: Predicate = always) -> SpectrumMatch:
        solutions = self.solutions_for(structure, threshold_fn=threshold_fn)
        if not solutions:
            raise KeyError(structure)
        best_score = -float('inf')
        best_match = None
        for sol in solutions:
            if best_score < sol.score:
                best_score = sol.score
                best_match = sol
        return best_match


class MassShiftDeconvolutionGraphEdge(ChromatogramGraphEdge):
    def __hash__(self):
        return hash(frozenset((self.node_a.index, self.node_b.index)))

    def __eq__(self, other):
        if frozenset((self.node_a.index, self.node_b.index)) == frozenset((other.node_a.index, other.node_b.index)):
            return self.transition == other.transition
        return False

    def __ne__(self, other):
        return not self == other


class MassShiftDeconvolutionGraph(ChromatogramGraph[MassShiftDeconvolutionGraphNode, MassShiftDeconvolutionGraphEdge]):
    node_cls = MassShiftDeconvolutionGraphNode
    edge_cls = MassShiftDeconvolutionGraphEdge

    mass_shifts: Set[MassShiftBase]

    def __init__(self, chromatograms: List['TandemAnnotatedChromatogram']):
        super(MassShiftDeconvolutionGraph, self).__init__(chromatograms)
        mass_shifts = set()
        for node in self:
            mass_shifts.update(node.mass_shifts)
        self.mass_shifts = mass_shifts

    @property
    def tandem_solutions(self):
        solution_sets = []
        for node in self.nodes:
            solution_sets.extend(node.tandem_solutions)
        return solution_sets

    def __iter__(self):
        return iter(self.nodes)

    def _construct_graph_nodes(self, chromatograms: List['TandemAnnotatedChromatogram']):
        nodes = []
        for i, chroma in enumerate(chromatograms):
            node = (self.node_cls(chroma, i))
            nodes.append(node)
            if node.chromatogram.composition:
                self.enqueue_seed(node)
                self.assignment_map[node.chromatogram.composition] = node
        return nodes

    def find_edges(self, node: MassShiftDeconvolutionGraphNode, query_width: float = 0.01, threshold_fn: Predicate = always,  **kwargs):
        query = TimeQuery(node.chromatogram, query_width)
        nodes: List[MassShiftDeconvolutionGraphNode] = self.rt_tree.overlaps(
            query.start, query.end)

        structure = node.chromatogram.structure

        for other in nodes:
            solutions = other.solutions_for(
                structure, threshold_fn=threshold_fn)
            if solutions:
                shifts_in_solutions = frozenset(
                    {m.mass_shift for m in solutions})
                self.edges.add(
                    self.edge_cls(node, other, (frozenset(node.mass_shifts), shifts_in_solutions)))

    def build(self, query_width: float = 0.01, threshold_fn: Predicate = always, **kwargs):
        for node in self.iterseeds():
            self.find_edges(node, query_width=query_width,
                            threshold_fn=threshold_fn, **kwargs)


class RepresenterDeconvolution(TaskBase):
    max_depth = 10
    threshold_fn: Predicate
    key_fn: Callable[[TargetType], Hashable]

    group: List[MassShiftDeconvolutionGraphNode]

    conflicted: DefaultDict[MassShiftDeconvolutionGraphNode, List[KeyTargetMassShift]]
    solved: DefaultDict[KeyTargetMassShift, List[MassShiftDeconvolutionGraphNode]]
    participants: DefaultDict[KeyTargetMassShift, List[MassShiftDeconvolutionGraphNode]]
    best_scores: DefaultDict[KeyMassShift, float]

    key_to_solutions: DefaultDict[KeyTargetMassShift, Set[TargetType]]

    spectrum_match_backfiller: Optional["SpectrumMatchBackFiller"]
    revision_validator: MS2RevisionValidator

    def __init__(self, group, threshold_fn=always, key_fn=build_glycopeptide_key,
                 spectrum_match_backfiller: Optional["SpectrumMatchBackFiller"]=None,
                 revision_validator: Optional[MS2RevisionValidator]=None):
        if revision_validator is None:
            revision_validator = MS2RevisionValidator(threshold_fn)
        self.group = group
        self.threshold_fn = threshold_fn
        self.key_fn = key_fn
        self.spectrum_match_backfiller = spectrum_match_backfiller
        self.revision_validator = revision_validator

        self.participants = None
        self.key_to_solutions = None
        self.conflicted = None
        self.solved = None
        self.best_scores = None

        self.build()

    def _simplify(self):
        # When there is an un-considered double shift, this could lead to
        # choosing that double-shift being called a trivial solution and
        # locking in a specific interpretation regardless of weight. So
        # this method won't be used automatically.

        # Search all members' assignment lists and see if any
        # can be easily assigned.
        for member, conflicts in list(self.conflicted.items()):
            # There is no conflict, trivially solved.
            if len(conflicts) == 1:
                self.solved[conflicts[0]].append(member)
                self.conflicted.pop(member)
            else:
                keys = {k for k, e, m in conflicts}
                # There are multiple structures, but they all share the same key, essentially
                # a positional isomer. Trivially solved, but need to use care when merging to avoid
                # combining the different isomers.
                if len(keys) == 1:
                    self.solved[conflicts[0]].append(member)
                    self.conflicted.pop(member)

    def build(self):
        '''Construct the initial graph, populating all the maps
        and then correcting for unsupported mass shifts.
        '''
        participants = DefaultDict(list)
        conflicted = DefaultDict(list)
        key_to_solutions = DefaultDict(set)
        best_scores = DefaultDict(float)

        for member in self.group:
            for part in member.most_representative_solutions(
                    threshold_fn=self.threshold_fn, reject_shifted=False, percentile_threshold=0.4):
                key = self.key_fn(part.solution)
                mass_shift = part.match.mass_shift
                score = part.best_score
                key_to_solutions[key].add(part.solution)
                participants[key, part.solution, mass_shift].append(member)
                best_scores[key, mass_shift] = max(score, best_scores[key, mass_shift])
                conflicted[member].append((key, part.solution, mass_shift))

            for part in member.most_representative_solutions(
                    threshold_fn=self.threshold_fn, reject_shifted=True, percentile_threshold=0.4):
                key = self.key_fn(part.solution)
                mass_shift = part.match.mass_shift
                score = part.best_score
                key_to_solutions[key].add(part.solution)
                best_scores[key, mass_shift] = max(score, best_scores[key, mass_shift])
                participants[key, part.solution, mass_shift].append(member)
                conflicted[member].append((key, part.solution, mass_shift))

        self.solved = DefaultDict(list)
        self.participants = participants
        self.conflicted = conflicted
        self.best_scores = best_scores
        self.key_to_solutions = key_to_solutions
        self.prune_unsupported_participants()

    def find_supported_participants(self) -> Dict[Tuple, bool]:
        has_unmodified = DefaultDict(bool)
        for (key, solution, mass_shift) in self.participants:
            if mass_shift == Unmodified:
                has_unmodified[key] = True
        return has_unmodified

    def prune_unsupported_participants(self, support_map: Dict[Tuple, bool] = None):
        '''Remove solutions that are not supported by an Unmodified
        condition except when there is *no alternative*.

        Uses :meth:`find_supported_participants` to label solutions.

        Returns
        -------
        pruned_conflicts : dict[list]
            A mapping from graph node to pruned solution
        restored : list
            A list of the solutions that would have been pruned
            but had to be restored to give a node support.
        '''

        if support_map is None:
            support_map = self.find_supported_participants()

        pruned_participants: Dict[KeyTargetMassShift, List[MassShiftDeconvolutionGraphNode]] = {}
        pruned_scores: Dict[Tuple[Hashable, MassShiftBase], float] = {}
        pruned_conflicts: DefaultDict[MassShiftDeconvolutionGraphNode,
                                      List[Tuple[Hashable, TargetType, MassShiftBase]]] = DefaultDict(list)
        pruned_solutions: List[TargetType] = []
        kept_solutions = []

        key: Hashable
        solution: TargetType
        mass_shift: MassShiftBase
        # Loop through the participants, removing links to unsupported solutions
        # and their metadata.
        for (key, solution, mass_shift), nodes in list(self.participants.items()):
            if not support_map[key]:
                q = (key, solution, mass_shift)
                try:
                    pruned_scores[key, mass_shift] = self.best_scores.pop((key, mass_shift))
                except KeyError:
                    pass
                pruned_participants[q] = self.participants.pop(q)
                pruned_solutions.append(solution)
                for node in nodes:
                    pruned_conflicts[node].append(q)
                    self.conflicted[node].remove(q)
            else:
                kept_solutions.append(solution)

        restored: List[MassShiftDeconvolutionGraphNode] = []
        # If all solutions for a given node have been removed, have to put them back.
        for node, options in list(self.conflicted.items()):
            if not options:
                alternatives = self.find_alternatives(
                    node, pruned_solutions)
                if alternatives:
                    for alt in alternatives:
                        solution = alt.target
                        mass_shift = alt.mass_shift
                        key = self.key_fn(solution)
                        q = key, solution, mass_shift

                        self.participants[q].append(node)
                        self.conflicted[node].append(q)
                        self.key_to_solutions[key].add(solution)
                        if self.threshold_fn(alt):
                            self.best_scores[key, mass_shift] = max(
                                alt.score, self.best_scores[key, mass_shift])
                        else:
                            self.best_scores[key, mass_shift] = 1e-3
                    continue

                restored.append(node)
                self.conflicted[node] = conflicts = pruned_conflicts.pop(node)
                for q in conflicts:
                    (key, solution, mass_shift) = q
                    try:
                        self.best_scores[key, mass_shift] = pruned_scores.pop(
                            (key, mass_shift))
                    except KeyError:
                        pass
                    self.participants[q].append(node)
        return pruned_conflicts, restored

    def find_alternatives(self, node: MassShiftDeconvolutionGraphNode, pruned_solutions, ratio_threshold=0.9):
        alternatives = {alt_solution for _,
                        alt_solution, _ in self.participants}
        ratios = []

        for solution in pruned_solutions:
            for sset in node.tandem_solutions:
                try:
                    sm = sset.solution_for(solution)
                    if not self.threshold_fn(sm):
                        continue
                    for alt in alternatives:
                        try:
                            sm2 = sset.solution_for(alt)
                            weight1 = sm.score
                            weight2 = sm2.score / weight1
                            if weight2 > ratio_threshold:
                                ratios.append((weight1, weight2, sm, sm2))
                        except KeyError:
                            continue
                except KeyError:
                    continue
        ratios.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if not ratios:
            return []
        weight1, weight2, sm, sm2 = ratios[0]
        kept = []
        for pair in ratios:
            if weight1 - pair[0] > 1e-5:
                break
            if weight2 - pair[1] > 1e-5:
                break
            kept.append(sm2)
        return kept

    def resolve(self):
        '''For each conflicted node, try to assign it to a solution based upon
        the set of all solved nodes.

        Returns
        -------
        changes : int
            The number of nodes assigned during this execution.
        '''
        tree: DefaultDict[Hashable, Dict[MassShiftBase, List[MassShiftDeconvolutionGraphNode]]] = DefaultDict(dict)
        # Build a tree of key->mass_shift->members
        for (key, solution, mass_shift), members in self.solved.items():
            tree[key][mass_shift] = list(members)

        changes = 0

        # For each conflicted member, search through all putative explanations of the
        # chromatogram and see if another mass shift state has been identified
        # for that explanation's key. If so, and that explanation spans this
        # member, then add the putative explanation to the set of hits with weight
        # equal to that best score (sum across all supporting mass shifts for same key)
        # for that putative explanation.
        # Sort for determinism.
        for member, conflicts in sorted(
                self.conflicted.items(),
                key=lambda x: x[0].chromatogram.total_signal, reverse=True):
            hits: DefaultDict[KeyTargetMassShift, float] = DefaultDict(float)
            for key, solution, mass_shift in conflicts:
                # Other solved groups' keys to solved mass shifts
                shifts_to_solved = tree[key]
                for solved_mass_shift, solved_members in shifts_to_solved.items():
                    if any(m.overlaps(member) for m in solved_members):
                        hits[key, solution, mass_shift] += self.best_scores[key, solved_mass_shift]
            if not hits:
                continue
            # Select the solution with the greatest total weight.
            ordered_options = sorted(hits.items(), key=lambda x: x[1], reverse=True)
            (best_key, best_solution, best_mass_shift), score = ordered_options[0]

            # Add a new solution to the tracker, and update the tree
            self.solved[best_key, best_solution, best_mass_shift].append(member)
            tree[best_key][best_mass_shift] = self.solved[best_key, best_solution, best_mass_shift]

            self.conflicted.pop(member)
            changes += 1
        return changes

    def total_weight_for_keys(self, score_map=None):
        if score_map is None:
            score_map = self.best_scores
        acc = DefaultDict(list)
        for (key, mass_shift), score in score_map.items():
            acc[key].append(score)
        return {
            key: sum(val) for key, val in acc.items()
        }

    def nominate_key_mass_shift(self, score_map=None):
        '''Find the key with the greatest total score
        over all mass shifts and nominate it and its
        best mass shift.

        Returns
        -------
        key : tuple
            The key tuple for the best solution
        mass_shift : :class:`~.MassShiftBase`
            The best mass shift for the best solution
        '''
        if score_map is None:
            score_map = self.best_scores
        totals = self.total_weight_for_keys(score_map)
        if len(totals) == 0:
            return None, None
        best_key, _ = max(totals.items(), key=lambda x: x[1])

        best_mass_shift = None
        best_score = -float('inf')
        for (key, mass_shift), score in score_map.items():
            if key == best_key:
                if best_score < score:
                    best_score = score
                    best_mass_shift = mass_shift
        return best_key, best_mass_shift

    def find_starting_point(self):
        '''Find a starting point for deconvolution and mark it as solved.

        Returns
        -------
        key : tuple
            The key tuple for the best solution
        best_match : object
            The target that belongs to this key-mass shift pair.
        mass_shift : :class:`~.MassShiftBase`
            The best mass shift for the best solution
        '''
        nominated_key, nominated_mass_shift = self.nominate_key_mass_shift()
        if nominated_key is None:
            return None, None, None
        best_node = best_match = None
        assignable = []

        for member, conflicts in self.conflicted.items():
            for key, solution, mass_shift in conflicts:
                if (key == nominated_key) and (mass_shift == nominated_mass_shift):
                    try:
                        match = member.best_match_for(solution, threshold_fn=self.threshold_fn)
                        if match is None:
                            continue
                        assignable.append((member, match))
                    except KeyError:
                        continue
        if not assignable:
            self.log("Could not find a solution matching %r %r. Trying to drop the mass shift." % (nominated_key, nominated_mass_shift))
            for member, conflicts in self.conflicted.items():
                for key, solution, mass_shift in conflicts:
                    self.log(key)
                    if (key == nominated_key):
                        self.log("Key Match %r for %r" % (key, solution))
                        try:
                            match = member.best_match_for(solution, threshold_fn=self.threshold_fn)
                            self.log("Match: %r" % match)
                            if match is None:
                                self.log("Match is None, skipping")
                                continue
                            assignable.append((member, match))
                        except KeyError as err:
                            self.log("KeyError, skipping: %r" % err)
                            continue


        if assignable:
            best_node, best_match = max(assignable, key=lambda x: x[1].score)

        self.solved[nominated_key, best_match.target, nominated_mass_shift].append(best_node)
        self.conflicted.pop(best_node)
        return nominated_key, best_match, nominated_mass_shift

    def recurse(self, depth=0):
        subgroup = list(self.conflicted)
        subprob = self.__class__(subgroup, threshold_fn=self.threshold_fn, key_fn=self.key_fn)
        subprob.solve(depth=depth + 1)

        for key, val in subprob.solved.items():
            self.solved[key].extend(val)

        n = len(self.conflicted)
        k = len(subprob.conflicted)
        changed = n - k
        self.conflicted = subprob.conflicted
        return changed

    def default(self):
        '''Assign each chromatogram to its highest scoring identification
        individually, the solution graph could not be deconvolved.
        '''
        for member, conflicts in list(self.conflicted.items()):
            entry = member.most_representative_solutions(threshold_fn=self.threshold_fn)[0]
            key = self.key_fn(entry.solution)
            self.solved[key, entry.solution, entry.match.mass_shift].append(member)
            self.conflicted.pop(member)

    def solve(self, depth=0):
        '''Deconvolve the graph, removing conflicts and adding nodes to the
        solved set.

        If no nodes are solved initially, :meth:`find_starting_point` is used to select one.

        Solutions are found using :meth:`resolve` to build on starting points and
        :meth:`recurse`

        '''
        if depth > self.max_depth:
            self.default()
            return self
        if not self.solved:
            nominated_key, _, _ = self.find_starting_point()
            if nominated_key is None:
                self.log("No starting point found in %r, defaulting.", self.group)
                self.default()
                return self

        i = 0
        while self.conflicted:
            changed = self.resolve()
            if changed == 0 and self.conflicted:
                changed = self.recurse(depth)
            if changed == 0:
                break
            i += 1
            if i > 20:
                break
        return self

    def _representer_best_scores_percentiles(self, member: 'TandemAnnotatedChromatogram', solutions: Set[TargetType]) -> Tuple[Dict[TargetType, SpectrumMatch], Dict[TargetType, float], Dict[TargetType, float]]:
        entries: Dict[TargetType, List[SpectrumMatch]] = dict()
        for sol in solutions:
            entries[sol] = member.solutions_for(
                sol, threshold_fn=self.threshold_fn)
        scores: Dict[TargetType, float] = {}
        total = 1e-6
        best_matches: Dict[TargetType, SpectrumMatch] = {}
        for k, v in entries.items():
            best_match: Optional[SpectrumMatch] = None
            sk: float = 0
            for match in v:
                if best_match is None or best_match.score < match.score:
                    best_match = match
                sk += match.score
            best_matches[k] = best_match
            scores[k] = sk
            total += sk
        percentiles = {k: v / total for k, v in scores.items()}
        return best_matches, scores, percentiles

    def _representer_solution_entries(self, member: 'TandemAnnotatedChromatogram',
                                      best_matches: Dict[TargetType, SpectrumMatch],
                                      scores: Dict[TargetType, float],
                                      percentiles: Dict[TargetType, float]) -> List[SolutionEntry]:
        sols = []
        for k, best_match in best_matches.items():
            score = scores[k]
            if best_match is None:
                try:
                    best_match = member.best_match_for(k)
                except KeyError:
                    continue
                score = best_match.score
            sols.append(
                SolutionEntry(
                    k, score, percentiles[k], best_match.score, best_match))

        sols = parsimony_sort(sols)
        return sols

    def assign_representers(self, percentile_threshold=1e-5):
        '''Apply solutions to solved graph nodes, constructing new chromatograms with
        corrected mass shifts, entities, and representative_solution attributes and then
        merge within keys as appropriate.

        Returns
        -------
        merged : list[ChromatogramWrapper]
            The merged chromatograms
        '''
        # Note suggested to switch to list, but reason unclear.
        assigned: DefaultDict[Any, Set['TandemAnnotatedChromatogram']] = DefaultDict(set)

        invalidated_alternatives: DefaultDict[Any, Set[TargetType]] = DefaultDict(set)

        # For each (key, mass shift) pair and their members, compute the set of representers for
        # that key from all structures that as subsumed into it (repeated peptide and alternative
        # localization).
        # NOTE: Alternative localizations fall under the same key, so we have to re-do ranking
        # of solutions again here to make sure that the aggregate scores are available to separate
        # different localizations. Alternatively, may need to port in localization-separating key
        # logic.
        _mass_shift: MassShiftBase
        for (key, _solution, _mass_shift), members in self.solved.items():
            # We do not use _mass_shift here because it will be inferred from the best solution
            # for `key` on the chromatogram anyway, we only needed it keep the hash distinct
            #
            # We look up solutions instead of using _solution from the loop here because
            # shared peptides between proteins need to be tracked separately.
            solutions = self.key_to_solutions[key]
            for member in members:
                # Summarize each explainer
                best_matches, scores, percentiles = self._representer_best_scores_percentiles(member, solutions)

                # Pack each explainer into a SolutionEntry instance
                sols = self._representer_solution_entries(member, best_matches, scores, percentiles)
                if sols:
                    # This difference is not using the absolute value to allow for scenarios where
                    # a worse percentile is located at position 0 e.g. when hoisting via parsimony.
                    representers = [x for x in sols if (
                        sols[0].percentile - x.percentile) < percentile_threshold]
                else:
                    representers = []

                # Mass shifts will be applied when we assign the representer
                fresh = member.chromatogram.clone().drop_mass_shifts()

                fresh.assign_entity(representers[0])
                fresh.representative_solutions = representers

                assigned[key].add(fresh)
                if key != self.key_fn(member.chromatogram.entity):
                    invalidated_alternatives[key].add(member.chromatogram.entity)

        merged = []
        for key, members in assigned.items():
            members: List['TandemAnnotatedChromatogram'] = sorted(members, key=lambda x: x.start_time)
            invalidated_targets = invalidated_alternatives[key]
            retained: List['TandemAnnotatedChromatogram'] = []

            member_targets: Set[TargetType] = set()
            for chrom in members:
                member_targets.add(chrom.entity)

            member_targets -= invalidated_targets

            can_merge: List['TandemAnnotatedChromatogram'] = []

            for member in members:
                if self.revision_validator.has_valid_matches(member, member_targets):
                    can_merge.append(member)
                else:
                    retained_solutions = member.most_representative_solutions(self.threshold_fn)
                    member.representative_solutions = retained_solutions
                    member.assign_entity(retained_solutions[0])
                    retained.append(member)

            sink = None
            if can_merge:
                sink = can_merge[0]
                did_update_sink_id = False
                if sink.entity not in member_targets:
                    self.error(f"... While deconvolving mass shifts, sink node {sink}'s label not in valid list {member_targets}")
                    options = sorted(
                        filter(
                            lambda x: x.solution in member_targets,
                            sink.compute_representative_weights(
                                self.threshold_fn)
                        ),
                        key=lambda x: x.score,
                        reverse=True
                    )
                    if options:
                        self.error(f"... Failed to re-assign {sink} to a valid target")
                        sink.assign_entity(options[0])
                    # May not be exhaustive enough, search for "best" case in `can_merge` matching
                    # to a `member_targets` before assigning `sink` role to a chromatogram?
                    did_update_sink_id = True

                for member in can_merge[1:]:
                    # Might this obscure the "best localization"? Perhaps, but localization is already getting
                    # mowed down at the aggregate level.
                    if self.revision_validator.can_rewrite_best_matches(member, sink.entity):
                        self.revision_validator.do_rewrite_best_matches(member, sink.entity, invalidated_targets)
                        sink.merge_in_place(member)
                    else:
                        retained_solutions = member.most_representative_solutions(self.threshold_fn)
                        member.representative_solutions = retained_solutions
                        member.assign_entity(retained_solutions[0])
                        retained.append(member)

                if did_update_sink_id:
                    options = sorted(
                        filter(
                            lambda x: x.solution in member_targets,
                            sink.compute_representative_weights(
                                self.threshold_fn)
                        ),
                        key=lambda x: x.score,
                        reverse=True
                    )
                    if options:
                        self.error(f"... Failed to re-assign {sink} to a valid target after merging!")
                        sink.assign_entity(options[0])

            if sink is not None:
                merged.append(sink)
            if retained:
                merged.extend(retained)
        return merged
