import time
from collections import defaultdict

import numpy as np

from glycan_profiling.task import TaskBase

from glycan_profiling.chromatogram_tree import (
    Unmodified,
    ChromatogramFilter,
    ChromatogramOverlapSmoother,
    prune_bad_adduct_branches)

from glycan_profiling.scoring import (
    ChromatogramSolution,
    ChromatogramScorer)

from glycan_profiling.composition_distribution_model import smooth_network, display_table


class ChromatogramEvaluator(TaskBase):
    acceptance_threshold = 0.4
    ignore_below = 1e-5

    def __init__(self, scoring_model=None):
        if scoring_model is None:
            scoring_model = ChromatogramScorer()
        self.scoring_model = scoring_model

    def configure(self, analysis_info):
        self.scoring_model.configure(analysis_info)

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        filtered = ChromatogramFilter.process(
            chromatograms, delta_rt=delta_rt, min_points=min_points)
        if smooth_overlap_rt:
            filtered = ChromatogramOverlapSmoother(filtered)
        solutions = []
        i = 0
        n = len(filtered)
        for case in filtered:
            start = time.time()
            i += 1
            if i % 1000 == 0:
                self.log("%0.2f%% chromatograms evaluated (%d/%d)" % (i * 100. / n, i, n))
            try:
                sol = self.evaluate_chromatogram(case)
                if self.scoring_model.accept(sol):
                    solutions.append(sol)
                else:
                    if sol.glycan_composition:
                        self.debug("Rejecting %s with score %s %s" % (
                            sol, sol.score, sol.score_components()))
                end = time.time()
                # Report on anything that took more than 30 seconds to evaluate
                if end - start > 30.0:
                    self.log("%r took a long time to evaluated (%0.2fs)" % (case, end - start))
            except (IndexError, ValueError):
                continue
        solutions = ChromatogramFilter(solutions)
        return solutions

    def evaluate_chromatogram(self, chromatogram):
        score_set = self.scoring_model.compute_scores(chromatogram)
        score = score_set.product()
        return ChromatogramSolution(
            chromatogram, score, scorer=self.scoring_model,
            score_set=score_set)

    def finalize_matches(self, solutions):
        out = []
        for sol in solutions:
            if sol.score <= self.ignore_below:
                continue
            elif (sol.composition is None) and (Unmodified not in sol.adducts):
                continue
            out.append(sol)
        solutions = ChromatogramFilter(out)
        return solutions

    def score(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
              adducts=None, *args, **kwargs):

        solutions = self.evaluate(
            chromatograms, delta_rt, min_points, smooth_overlap_rt, *args, **kwargs)

        if adducts:
            hold = self.prune_adducts(solutions)
            self.log("Re-evaluating after adduct pruning")
            solutions = self.evaluate(hold, delta_rt, min_points, smooth_overlap_rt,
                                      *args, **kwargs)

        solutions = self.finalize_matches(solutions)
        return solutions

    def prune_adducts(self, solutions):
        return prune_bad_adduct_branches(ChromatogramFilter(solutions))

    def acceptance_filter(self, solutions, threshold=None):
        if threshold is None:
            threshold = self.acceptance_threshold
        return ChromatogramFilter([
            sol for sol in solutions
            if sol.score >= threshold and not sol.used_as_adduct
        ])

    def update_parameters(self, param_dict):
        param_dict['scoring_model'] = self.scoring_model


class LogitSumChromatogramEvaluator(ChromatogramEvaluator):
    acceptance_threshold = 4
    ignore_below = 2

    def __init__(self, scoring_model):
        super(LogitSumChromatogramEvaluator, self).__init__(scoring_model)

    def prune_adducts(self, solutions):
        return prune_bad_adduct_branches(ChromatogramFilter(solutions), score_margin=2.5)

    def evaluate_chromatogram(self, chromatogram):
        score_set = self.scoring_model.compute_scores(chromatogram)
        logitsum_score = score_set.logitsum()
        return ChromatogramSolution(
            chromatogram, logitsum_score, scorer=self.scoring_model,
            score_set=score_set)

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        solutions = super(LogitSumChromatogramEvaluator, self).evaluate(
            chromatograms, delta_rt=delta_rt, min_points=min_points,
            smooth_overlap_rt=smooth_overlap_rt, *args, **kwargs)

        accumulator = defaultdict(list)
        for case in solutions:
            accumulator[case.key].append(case)
        solutions = []
        for group, members in accumulator.items():
            members = sorted(members, key=lambda x: x.score, reverse=True)
            reference = members[0]
            base = reference.clone()
            for other in members[1:]:
                base = base.merge(other)
            merged = reference.__class__(
                base, reference.score, scorer=reference.scorer,
                score_set=reference.score_set)
            solutions.append(merged)
        return ChromatogramFilter(solutions)


class LaplacianRegularizedChromatogramEvaluator(LogitSumChromatogramEvaluator):
    def __init__(self, scoring_model, network, smoothing_factor=None, grid_smoothing_max=1.0,
                 regularization_model=None):
        super(LaplacianRegularizedChromatogramEvaluator,
              self).__init__(scoring_model)
        self.network = network
        self.smoothing_factor = smoothing_factor
        self.grid_smoothing_max = grid_smoothing_max
        self.regularization_model = regularization_model

    def evaluate(self, chromatograms, delta_rt=0.25, min_points=3, smooth_overlap_rt=True,
                 *args, **kwargs):
        solutions = super(LaplacianRegularizedChromatogramEvaluator, self).evaluate(
            chromatograms, delta_rt=delta_rt, min_points=min_points,
            smooth_overlap_rt=smooth_overlap_rt, *args, **kwargs)
        self.log("... Applying Network Smoothing Regularization")
        updated_network, search, params = smooth_network(
            self.network, solutions, lmbda=self.smoothing_factor,
            lambda_max=self.grid_smoothing_max,
            model_state=self.regularization_model)
        solutions = sorted(solutions, key=lambda x: x.score, reverse=True)
        # TODO - Use aggregation across multiple observations for the same glycan composition
        # instead of discarding all but the top scoring feature?
        seen = dict()
        unannotated = []
        for sol in solutions:
            if sol.glycan_composition is None:
                unannotated.append(sol)
                continue
            if sol.glycan_composition in seen:
                continue
            seen[sol.glycan_composition] = sol
            node = updated_network[sol.glycan_composition]
            if sol.score > self.acceptance_threshold:
                sol.score = node.score
            else:
                # Do not permit network smoothing to boost scores below acceptance_threshold
                if node.score < sol.score:
                    sol.score = node.score
        self.network_parameters = params
        self.grid_search = search
        display_table(
            search.model.neighborhood_names,
            np.array(params.tau).reshape((-1, 1)),
            print_fn=lambda x: self.log("...... %s" % (x,)))
        self.log("...... smoothing factor: %0.3f; threshold: %0.3f" % (
            params.lmbda, params.threshold))
        return ChromatogramFilter(list(seen.values()) + unannotated)

    def update_parameters(self, param_dict):
        super(LaplacianRegularizedChromatogramEvaluator, self).update_parameters(param_dict)
        param_dict['network_parameters'] = self.network_parameters
        param_dict['network_model'] = self.grid_search
