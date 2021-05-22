import os

from collections import defaultdict, namedtuple
import array
import numpy as np

from glycan_profiling.task import TaskBase, log_handle
from glycan_profiling.chromatogram_tree import (
    ChromatogramWrapper, build_rt_interval_tree, ChromatogramFilter,
    Unmodified)

from glycan_profiling.chromatogram_tree.chromatogram import GlycopeptideChromatogram
from glycan_profiling.chromatogram_tree.relation_graph import (
    ChromatogramGraph, ChromatogramGraphEdge, ChromatogramGraphNode, TimeQuery)

from .spectrum_match.solution_set import NOParsimonyMixin


class MassShiftDeconvolutionGraphNode(ChromatogramGraphNode):
    def __init__(self, chromatogram, index, edges=None):
        super(MassShiftDeconvolutionGraphNode, self).__init__(
            chromatogram, index, edges=edges)

    def most_representative_solutions(self, threshold_fn=lambda x: True, reject_shifted=False, percentile_threshold=1e-5):
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

    def solutions_for(self, structure, threshold_fn=lambda x: True, reject_shifted=False):
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

    def best_match_for(self, structure, threshold_fn=lambda x: True):
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


class MassShiftDeconvolutionGraph(ChromatogramGraph):
    node_cls = MassShiftDeconvolutionGraphNode
    edge_cls = MassShiftDeconvolutionGraphEdge

    def __init__(self, chromatograms):
        super(MassShiftDeconvolutionGraph, self).__init__(chromatograms)
        mass_shifts = set()
        for node in self:
            mass_shifts.update(node.mass_shifts)
        self.mass_shifts = mass_shifts

    def __iter__(self):
        return iter(self.nodes)

    def _construct_graph_nodes(self, chromatograms):
        nodes = []
        for i, chroma in enumerate(chromatograms):
            node = (self.node_cls(chroma, i))
            nodes.append(node)
            if node.chromatogram.composition:
                self.enqueue_seed(node)
                self.assignment_map[node.chromatogram.composition] = node
        return nodes

    def find_edges(self, node, query_width=0.01, threshold_fn=lambda x: True,  **kwargs):
        query = TimeQuery(node.chromatogram, query_width)
        nodes = self.rt_tree.overlaps(query.start, query.end)

        structure = node.chromatogram.structure

        for other in nodes:
            solutions = other.solutions_for(
                structure, threshold_fn=threshold_fn)
            if solutions:
                shifts_in_solutions = frozenset(
                    {m.mass_shift for m in solutions})
                self.edges.add(
                    self.edge_cls(node, other, (frozenset(node.mass_shifts), shifts_in_solutions)))

    def build(self, query_width=0.01, threshold_fn=lambda x: True, **kwargs):
        for node in self.iterseeds():
            self.find_edges(node, query_width=query_width,
                            threshold_fn=threshold_fn, **kwargs)


SolutionEntry = namedtuple("SolutionEntry", "solution, score, percentile, best_score, match")

debug_mode = bool(os.environ.get('GLYCRESOFTDEBUG', False))


class NOParsimonyRepresentativeSelector(NOParsimonyMixin):
    def get_score(self, solution):
        return solution.percentile

    def get_target(self, solution):
        return solution.match.target

    def sort(self, solution_set):
        solution_set = sorted(solution_set, key=lambda x: x.percentile, reverse=True)
        try:
            if solution_set and self.get_target(solution_set[0]).is_o_glycosylated():
                solution_set = self.hoist_equivalent_n_linked_solution(solution_set)
        except AttributeError:
            import warnings
            warnings.warn("Could not determine glycosylation state of target of type %r" % type(self.get_target(solution_set[0])))
        return solution_set

    def __call__(self, solution_set):
        return self.sort(solution_set)


parsimony_sort = NOParsimonyRepresentativeSelector()


class RepresenterSelectionStrategy(object):
    def compute_weights(self, collection, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None):
        raise NotImplementedError()

    def select(self, representers):
        raise NotImplementedError()

    def sort_solutions(self, representers):
        return parsimony_sort(representers)

    def get_solutions_for_spectrum(self, solution_set, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None):
        return solution_set.get_top_solutions(
            d=5, reject_shifted=reject_shifted, targets_ignored=targets_ignored)

    def __call__(self, collection, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None):
        return self.compute_weights(
            collection, threshold_fn=threshold_fn, reject_shifted=reject_shifted, targets_ignored=targets_ignored)


class TotalBestRepresenterStrategy(RepresenterSelectionStrategy):
    def compute_weights(self, collection, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None):
        scores = defaultdict(float)
        best_scores = defaultdict(float)
        best_spectrum_match = dict()
        for psm in collection.tandem_solutions:
            if threshold_fn(psm):
                for sol in self.get_solutions_for_spectrum(psm, reject_shifted=reject_shifted, targets_ignored=targets_ignored):
                    if not threshold_fn(sol):
                        continue
                    if reject_shifted and sol.mass_shift != Unmodified:
                        continue
                    scores[sol.target] += (sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
                        best_spectrum_match[sol.target] = sol
        total = sum(scores.values())
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k],
                          best_spectrum_match[k]) for k, v in scores.items()
            if k in best_spectrum_match
        ]
        weights = self.sort_solutions(weights)
        return weights


class TotalAboveAverageBestRepresenterStrategy(RepresenterSelectionStrategy):
    def compute_weights(self, collection, threshold_fn=lambda x:True, reject_shifted=False, targets_ignored=None):
        scores = defaultdict(lambda: array.array('d'))
        best_scores = defaultdict(float)
        best_spectrum_match = dict()
        for psm in collection.tandem_solutions:
            if threshold_fn(psm):
                for sol in self.get_solutions_for_spectrum(psm, reject_shifted=reject_shifted, targets_ignored=targets_ignored):
                    if not threshold_fn(sol):
                        continue
                    if reject_shifted and sol.mass_shift != Unmodified:
                        continue
                    scores[sol.target].append(sol.score)
                    if best_scores[sol.target] < sol.score:
                        best_scores[sol.target] = sol.score
                        best_spectrum_match[sol.target] = sol
        population = np.concatenate(list(scores.values()))
        min_score = np.mean(population) - np.std(population)
        scores = {k: np.array(v) for k, v in scores.items()}
        thresholded_scores = {k: v[v >= min_score].sum() for k, v in scores.items()}

        total = sum([v for v in thresholded_scores.values()])
        weights = [
            SolutionEntry(k, v, v / total, best_scores[k],
                          best_spectrum_match[k]) for k, v in thresholded_scores.items()
            if k in best_spectrum_match
        ]
        weights = self.sort_solutions(weights)
        return weights


class SpectrumMatchSolutionCollectionBase(object):
    def compute_representative_weights(self, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None, strategy=None):
        """Calculate a total score for all matched structures across all time points for this
        solution collection, and rank them.

        This total score is the sum of the score over all spectrum matches for which that
        structure was the best match. The percentile is the ratio of the total score for the
        `i`th structure divided by the sum of total scores over all structures.

        Parameters
        ----------
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False

        Returns
        -------
        list
            A list of solutions, ranked by percentile.
        """
        weights = strategy(self)
        return weights



    def most_representative_solutions(self, threshold_fn=lambda x: True, reject_shifted=False, targets_ignored=None,
                                      percentile_threshold=1e-5):
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
        weights = self.compute_representative_weights(
            threshold_fn, reject_shifted=reject_shifted, targets_ignored=targets_ignored)
        if weights:
            # This difference is not using the absolute value to allow for scenarios where
            # a worse percentile is located at position 0 e.g. when hoisting via parsimony.
            representers = [x for x in weights if (
                weights[0].percentile - x.percentile) < percentile_threshold]
            return representers
        else:
            return []

    def solutions_for(self, structure, threshold_fn=lambda x: True, reject_shifted=False):
        '''Get all spectrum matches in this collection for a given
        structure.

        Parameters
        ----------
        structure : Hashable
            The structure collect matches for.
        threshold_fn: Callable
            A function that filters out invalid solutions based on some criteria, e.g.
            not passing the FDR threshold.
        reject_shifted: bool
            Whether or not to omit any solution that was not mass-shifted. Defaults to False

        Returns
        -------
        list
        '''
        solutions = []
        for sset in self.tandem_solutions:
            try:
                psm = sset.solution_for(structure)
                if threshold_fn(psm):
                    if psm.mass_shift != Unmodified and reject_shifted:
                        continue
                    solutions.append(psm)
            except KeyError:
                continue
        return solutions

    def best_match_for(self, structure, threshold_fn=lambda x: True):
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


class TandemAnnotatedChromatogram(ChromatogramWrapper, SpectrumMatchSolutionCollectionBase):
    def __init__(self, chromatogram):
        super(TandemAnnotatedChromatogram, self).__init__(chromatogram)
        self.tandem_solutions = []
        self.time_displaced_assignments = []
        self.best_msms_score = None
        self.representative_solutions = None

    def bisect_charge(self, charge):
        new_charge, new_no_charge = map(self.__class__, self.chromatogram.bisect_charge(charge))
        for hit in self.tandem_solutions:
            if hit.precursor_information.charge == charge:
                new_charge.add_solution(hit)
            else:
                new_no_charge.add_solution(hit)
        return new_charge, new_no_charge

    def bisect_mass_shift(self, mass_shift):
        new_mass_shift, new_no_mass_shift = map(self.__class__, self.chromatogram.bisect_mass_shift(mass_shift))
        for hit in self.tandem_solutions:
            if hit.best_solution().mass_shift == mass_shift:
                new_mass_shift.add_solution(hit)
            else:
                new_no_mass_shift.add_solution(hit)
        return new_mass_shift, new_no_mass_shift

    def split_sparse(self, delta_rt=1.0):
        parts = list(map(self.__class__, self.chromatogram.split_sparse(delta_rt)))
        for hit in self.tandem_solutions:
            nearest = None
            nearest_time = None
            time = hit.scan_time
            for part in parts:
                time_err = min(abs(part.start_time - time), abs(part.end_time - time))
                if time_err < nearest_time:
                    nearest = part
                    nearest_time = time_err
                if part.spans_time_point(time):
                    part.add_solution(hit)
                    break
            else:
                nearest.add_solution(hit)
        return parts

    def add_solution(self, item):
        case_mass = item.precursor_information.neutral_mass
        if abs(case_mass - self.chromatogram.neutral_mass) > 100:
            log_handle.log("Warning, mis-assigned spectrum match to chromatogram %r, %r" % (self, item))
        self.tandem_solutions.append(item)

    def add_displaced_solution(self, item):
        self.add_solution(item)

    def clone(self):
        new = super(TandemAnnotatedChromatogram, self).clone()
        new.tandem_solutions = list(self.tandem_solutions)
        new.time_displaced_assignments = list(self.time_displaced_assignments)
        new.best_msms_score = self.best_msms_score
        return new

    def merge(self, other):
        new = self.__class__(self.chromatogram.merge(other.chromatogram))
        new.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        new.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments
        return new

    def merge_in_place(self, other):
        new = self.chromatogram.merge(other.chromatogram)
        self.chromatogram = new
        self.tandem_solutions = self.tandem_solutions + other.tandem_solutions
        self.time_displaced_assignments = self.time_displaced_assignments + other.time_displaced_assignments

    def assign_entity(self, solution_entry, entity_chromatogram_type=GlycopeptideChromatogram):
        entity_chroma = entity_chromatogram_type(
            None,
            self.chromatogram.nodes, self.chromatogram.mass_shifts,
            self.chromatogram.used_as_mass_shift)
        entity_chroma.entity = solution_entry.solution
        if solution_entry.match.mass_shift != Unmodified:
            identified_shift = solution_entry.match.mass_shift
            for node in entity_chroma.nodes.unspool():
                if node.node_type == Unmodified:
                    node.node_type = identified_shift
                else:
                    node.node_type = (node.node_type + identified_shift)
            entity_chroma.invalidate()
        self.chromatogram = entity_chroma
        self.best_msms_score = solution_entry.best_score


class TandemSolutionsWithoutChromatogram(SpectrumMatchSolutionCollectionBase):
    def __init__(self, entity, tandem_solutions):
        self.entity = entity
        self.composition = entity
        self.tandem_solutions = tandem_solutions

    @classmethod
    def aggregate(cls, solutions):
        collect = defaultdict(list)
        for solution in solutions:
            best_match = solution.best_solution()
            collect[best_match.target.id].append(solution)
        out = []
        for group in collect.values():
            solution = group[0]
            best_match = solution.best_solution()
            structure = best_match.target
            out.append(cls(structure, group))
        return out


class ScanTimeBundle(object):
    def __init__(self, solution, scan_time):
        self.solution = solution
        self.scan_time = scan_time

    @property
    def score(self):
        try:
            return self.solution.score
        except AttributeError:
            return None

    def __hash__(self):
        return hash((self.solution, self.scan_time))

    def __eq__(self, other):
        return self.solution == other.solution and self.scan_time == other.scan_time

    def __repr__(self):
        return "ScanTimeBundle(%s, %0.4f, %0.4f)" % (
            self.solution.scan.id, self.score, self.scan_time)


def build_glycopeptide_key(solution):
    structure = solution
    key = (str(structure.clone().deglycosylate()), str(structure.glycan_composition))
    return key


def drop_mass_shifts(self):
    for node in self.nodes.unspool():
        node.node_type = Unmodified
    return self


class RepresenterDeconvolution(object):
    max_depth = 10

    def __init__(self, group, threshold_fn=lambda x: True, key_fn=build_glycopeptide_key):
        self.group = group
        self.threshold_fn = threshold_fn
        self.key_fn = key_fn

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
        participants = defaultdict(list)
        conflicted = defaultdict(list)
        key_to_solutions = defaultdict(set)
        best_scores = defaultdict(float)

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

        self.solved = defaultdict(list)
        self.participants = participants
        self.conflicted = conflicted
        self.best_scores = best_scores
        self.key_to_solutions = key_to_solutions
        self.prune_unsupported_participants()

    def find_supported_participants(self):
        has_unmodified = defaultdict(bool)
        for (key, solution, mass_shift) in self.participants:
            if mass_shift == Unmodified:
                has_unmodified[key] = True
        return has_unmodified

    def prune_unsupported_participants(self, support_map=None):
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

        pruned_participants = {}
        pruned_scores = {}
        pruned_conflicts = defaultdict(list)
        pruned_solutions = []
        kept_solutions = []

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

        restored = []
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
                        self.best_scores[key, mass_shift] = max(
                            alt.score, self.best_scores[key, mass_shift])
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

    def find_alternatives(self, node, pruned_solutions, ratio_threshold=0.9):
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
        tree = defaultdict(dict)
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
            print("... Resolving %r" % member)
            hits = defaultdict(float)
            for key, solution, mass_shift in conflicts:
                # Other solved groups' keys to solved mass shifts
                shifts_to_solved = tree[key]
                for solved_mass_shift, solved_members in shifts_to_solved.items():
                    if any(m.overlaps(member) for m in solved_members):
                        hits[key, solution, mass_shift] += self.best_scores[key, solved_mass_shift]
            if not hits:
                print("... No Hits")
                continue
            # Select the solution with the greatest total weight.
            ordered_options = sorted(hits.items(), key=lambda x: x[1], reverse=True)
            (best_key, best_solution, best_mass_shift), score = ordered_options[0]
            print("... Best solution", best_key, best_solution, best_mass_shift)

            # Add a new solution to the tracker, and update the tree
            self.solved[best_key, best_solution, best_mass_shift].append(member)
            tree[best_key][best_mass_shift] = self.solved[best_key, best_solution, best_mass_shift]

            self.conflicted.pop(member)
            changes += 1
        return changes

    def total_weight_for_keys(self, score_map=None):
        if score_map is None:
            score_map = self.best_scores
        acc = defaultdict(list)
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
        best_node = best_match = None
        assignable = []

        for member, conflicts in self.conflicted.items():
            for key, solution, mass_shift in conflicts:
                if (key == nominated_key) and (mass_shift == nominated_mass_shift):
                    match = member.best_match_for(solution, threshold_fn=self.threshold_fn)
                    if match is None:
                        continue
                    assignable.append((member, match))

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
            self.find_starting_point()

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

    def assign_representers(self, percentile_threshold=1e-5):
        '''Apply solutions to solved graph nodes, constructing new chromatograms with
        corrected mass shifts, entities, and representative_solution attributes and then
        merge within keys as appropriate.

        Returns
        -------
        merged : list[ChromatogramWrapper]
            The merged chromatograms
        '''
        # TODO: Switch to list
        assigned = defaultdict(set)

        # For each (key, mass shift) pair and their members, compute the set of representers for
        # that key from all structures that as subsumed into it (repeated peptide and alternative
        # localization).
        # NOTE: Alternative localizations fall under the same key, so we have to re-do ranking
        # of solutions again here to make sure that the aggregate scores are available to separate
        # different localizations. Alternatively, may need to port in
        for (key, solution, mass_shift), members in self.solved.items():
            solutions = self.key_to_solutions[key]
            for member in members:
                entries = dict()
                for sol in solutions:
                    entries[sol] = member.solutions_for(sol, threshold_fn=self.threshold_fn)
                scores = {}
                total = 0.0
                best_matches = {}
                for k, v in entries.items():
                    best_match = None
                    sk = 0
                    for match in v:
                        if best_match is None or best_match.score < match.score:
                            best_match = match
                        sk += match.score
                    best_matches[k] = best_match
                    scores[k] = sk
                    total += sk

                percentiles = {k: v / total for k, v in scores.items()}
                sols = []
                for k, v in best_matches.items():
                    sols.append(
                        SolutionEntry(
                            k, scores[k], percentiles[k], best_matches[k].score, best_matches[k]))
                sols = parsimony_sort(sols)
                if sols:
                    # This difference is not using the absolute value to allow for scenarios where
                    # a worse percentile is located at position 0 e.g. when hoisting via parsimony.
                    representers = [x for x in sols if (
                        sols[0].percentile - x.percentile) < percentile_threshold]
                else:
                    representers = []

                # TODO: Replace this with a method call
                fresh = drop_mass_shifts(member.chromatogram.clone())

                fresh.assign_entity(representers[0])
                fresh.representative_solutions = representers

                # TODO: switch to list.append
                assigned[key].add(fresh)

        merged = []
        for key, members in assigned.items():
            members = sorted(members, key=lambda x: x.start_time)
            sink = members[0]
            for member in members[1:]:
                sink.merge_in_place(member)
            merged.append(sink)
        return merged


class AnnotatedChromatogramAggregator(TaskBase):
    def __init__(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                threshold_fn=lambda x: x.q_value < 0.05):
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

    def aggregate_by_annotation(self, annotated_chromatograms):
        finished = []
        aggregated = defaultdict(list)
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

    def aggregate(self, annotated_chromatograms):
        self.log("Aggregating Common Entities: %d chromatograms" % (len(annotated_chromatograms,)))
        finished, aggregated = self.aggregate_by_annotation(annotated_chromatograms)
        finished.extend(self.combine_chromatograms(aggregated))
        self.log("After Merging: %d chromatograms" % (len(finished),))
        return finished

    def replace_solution(self, chromatogram, solutions):
        # select the best solution
        solutions = sorted(
                solutions, key=lambda x: x.score, reverse=True)

        # remove the invalidated mass shifts so that we're dealing with the raw masses again.
        current_shifts = chromatogram.chromatogram.mass_shifts
        if len(current_shifts) > 1:
            self.log("... Found a multiply shifted identification with no Unmodified state: %s" % (chromatogram.entity, ))
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
            accumulated_chromatogram = accumulated_chromatogram.merge(partition)
        chromatogram.chromatogram = accumulated_chromatogram

        # update the tandem annotations
        for solution_set in chromatogram.tandem_solutions:
            solution_set.mark_top_solutions(reject_shifted=True)
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
                    chromatogram = self.replace_solution(chromatogram, solutions)
                    self.debug("... Replacing %s with %s", original_entity, chromatogram.entity)
                    out.append(chromatogram)
                else:
                    self.log(
                        "... Could not find an alternative option for %r" % (chromatogram,))
                    out.append(chromatogram)
            else:
                out.append(chromatogram)
        return out

    def run(self):
        # merged_chromatograms = self.aggregate(self.annotated_chromatograms)
        # if self.require_unmodified:
        #     spliced = self.reassign_modified_only_cases(merged_chromatograms)
        #     merged_chromatograms = self.aggregate(spliced)
        # result = ChromatogramFilter(merged_chromatograms)
        # return result
        return ChromatogramFilter(self.annotated_chromatograms)


def aggregate_by_assigned_entity(annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                                 threshold_fn=lambda x: x.q_value < 0.05):
    job = AnnotatedChromatogramAggregator(
        annotated_chromatograms, delta_rt=delta_rt,
        require_unmodified=require_unmodified, threshold_fn=threshold_fn)
    finished = job.run()
    return finished


class ChromatogramMSMSMapper(TaskBase):
    def __init__(self, chromatograms, error_tolerance=1e-5, scan_id_to_rt=lambda x: x):
        self.chromatograms = ChromatogramFilter(map(
            TandemAnnotatedChromatogram, chromatograms))
        self.rt_tree = build_rt_interval_tree(self.chromatograms)
        self.scan_id_to_rt = scan_id_to_rt
        self.orphans = []
        self.error_tolerance = error_tolerance

    def find_chromatogram_spanning(self, time):
        return ChromatogramFilter([interv[0] for interv in self.rt_tree.contains_point(time)])

    def find_chromatogram_for(self, solution):
        try:
            precursor_scan_time = self.scan_id_to_rt(
                solution.precursor_information.precursor_scan_id)
        except Exception:
            precursor_scan_time = self.scan_id_to_rt(solution.scan_id)
        overlapping_chroma = self.find_chromatogram_spanning(precursor_scan_time)
        chromas = overlapping_chroma.find_all_by_mass(
            solution.precursor_information.neutral_mass, self.error_tolerance)
        if len(chromas) == 0:
            if debug_mode:
                self.log("... %s is an orphan" % (solution, ))
            self.orphans.append(ScanTimeBundle(solution, precursor_scan_time))
        else:
            if len(chromas) > 1:
                chroma = max(chromas, key=lambda x: x.total_signal)
            else:
                chroma = chromas[0]
            if debug_mode:
                self.log("... Assigning %s to %s %s (%0.2f -> %0.2f -> %0.2f)" % (
                    solution, chroma, chroma.spans_time_point(precursor_scan_time),
                    chroma.start_time, precursor_scan_time, chroma.end_time))
            chroma.tandem_solutions.append(solution)

    def assign_solutions_to_chromatograms(self, solutions):
        n = len(solutions)
        for i, solution in enumerate(solutions):
            if i % 1000 == 0:
                self.log("... %d/%d Solutions Handled (%0.2f%%)" % (i, n, (i * 100.0 / n)))
            self.find_chromatogram_for(solution)

    def distribute_orphans(self, threshold_fn=lambda x: x.q_value < 0.05):
        lost = []
        n = len(self.orphans)
        n_chromatograms = len(self.chromatograms)
        for j, orphan in enumerate(self.orphans):
            mass = orphan.solution.precursor_ion_mass
            time = orphan.scan_time
            if j % 100 == 0:
                self.log("... %r %d/%d Orphans Handled (%0.2f%%)" % (orphan, j, n, (j * 100.0 / n)))
            candidates = self.chromatograms.find_all_by_mass(mass, self.error_tolerance)
            if len(candidates) > 0:
                best_index = 0
                best_distance = float('inf')
                for i, candidate in enumerate(candidates):
                    dist = min(abs(candidate.start_time - time), abs(candidate.end_time - time))
                    if dist < best_distance:
                        best_index = i
                        best_distance = dist
                new_owner = candidates[best_index]
                if debug_mode:
                    self.log("... Assigning %r to %r with %d existing solutions with distance %0.3f" %
                             (orphan, new_owner, len(new_owner.tandem_solutions), best_distance))
                new_owner.add_displaced_solution(orphan.solution)
            else:
                if threshold_fn(orphan.solution):
                    if n_chromatograms > 0:
                        self.log("No chromatogram found for %r, q-value %0.4f (mass: %0.4f, time: %0.4f)" % (
                            orphan, orphan.solution.q_value, mass, time))
                    lost.append(orphan.solution)
        self.orphans = TandemSolutionsWithoutChromatogram.aggregate(lost)

    def assign_entities(self, threshold_fn=lambda x: x.q_value < 0.05, entity_chromatogram_type=None):
        if entity_chromatogram_type is None:
            entity_chromatogram_type = GlycopeptideChromatogram
        for chromatogram in self:
            solutions = chromatogram.most_representative_solutions(threshold_fn)
            if solutions:
                solutions = sorted(solutions, key=lambda x: x.score, reverse=True)
                if debug_mode:
                    self.log("... Assigning %s to %s out of %r\n" % (
                        solutions[0], chromatogram, solutions))
                chromatogram.assign_entity(solutions[0], entity_chromatogram_type=entity_chromatogram_type)
                chromatogram.representative_solutions = solutions

    def merge_common_entities(self, annotated_chromatograms, delta_rt=0.25, require_unmodified=True,
                              threshold_fn=lambda x: x.q_value < 0.05):
        job = AnnotatedChromatogramAggregator(
            annotated_chromatograms, delta_rt=delta_rt, require_unmodified=require_unmodified,
            threshold_fn=threshold_fn)
        result = job.run()
        return result

    def __len__(self):
        return len(self.chromatograms)

    def __iter__(self):
        return iter(self.chromatograms)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.chromatograms[i]
        else:
            return [self.chromatograms[j] for j in i]
