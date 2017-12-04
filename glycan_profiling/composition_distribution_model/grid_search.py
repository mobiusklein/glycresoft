from collections import OrderedDict, namedtuple
import re

import numpy as np
from matplotlib import pyplot as plt

from ms_deisotope.feature_map.profile_transform import peak_indices

from .constants import DEFAULT_RHO


class NetworkReduction(object):

    def __init__(self, store=None):
        if store is None:
            store = OrderedDict()
        self.store = store

    def getkey(self, key):
        return self.store[key]

    def getindex(self, ix):
        return self.getkey(list(self.store.keys())[ix])

    def searchkey(self, value):
        array = list(self.store.keys())
        ix = self.binsearch(array, value)
        key = array[ix]
        assert abs(value - key) < 1e-3
        return self.getkey(key)

    def put(self, key, value):
        self.store[key] = value

    def __getitem__(self, key):
        return self.getkey(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __iter__(self):
        return iter(self.store.values())

    @staticmethod
    def binsearch(array, value):
        lo = 0
        hi = len(array) - 1
        while hi - lo:
            i = (hi + lo) / 2
            x = array[i]
            if x == value:
                return i
            elif hi - lo == 1:
                return i
            elif x < value:
                lo = i
            elif x > value:
                hi = i
        return i

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        x = self.store.keys()
        y = [v.optimal_lambda for v in self.store.values()]
        ax.plot(x, y, **kwargs)
        xbound = ax.get_xlim()
        ybound = ax.get_ylim()
        ax.scatter(x, y)
        ax.set_xlim(*xbound)
        ax.set_ylim(*ybound)
        ax.set_xlabel("$S_o$ Threshold", fontsize=18)
        ax.set_ylabel("Optimal $\lambda$", fontsize=18)
        return ax

    def minimum_threshold_for_lambda(self, lmbda_target):
        best = None
        for value in reversed(self.store.values()):
            if best is None:
                best = value
                continue
            if abs(best.optimal_lambda - lmbda_target) >= abs(value.optimal_lambda - lmbda_target):
                if value.threshold < best.threshold:
                    best = value
        return best

    def press_weighted_mean_threshold(self):
        vals = list(self)
        return np.average(np.array(
            [v.threshold for v in vals]), weights=np.array(
            [v.press_residuals.min() for v in vals]))

    def keys(self):
        return self.store.keys()


class NetworkTrimmingSearchSolution(object):

    def __init__(self, threshold, lambda_values, press_residuals, network, observed=None,
                 updated=None, taus=None, model=None):
        self.threshold = threshold
        self.lambda_values = lambda_values
        self.press_residuals = press_residuals
        self.network = network
        self.taus = taus
        optimal_ix = np.argmin(self.press_residuals)
        self.optimal_lambda = self.lambda_values[optimal_ix]
        self.optimal_tau = self.taus[optimal_ix]
        self.minimum_residuals = self.press_residuals.min()
        self.observed = observed
        self.updated = updated
        self.model = model

    @property
    def n_kept(self):
        return len(self.network)

    @property
    def n_edges(self):
        return len(self.network.edges)

    @property
    def opt_lambda(self):
        return self.optimal_lambda

    def __repr__(self):
        min_press = min(self.press_residuals)
        opt_lambda = self.optimal_lambda
        return "NetworkTrimmingSearchSolution(%f, %d, %0.3f -> %0.3e)" % (
            self.threshold, self.n_kept, opt_lambda, min_press)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.lambda_values, self.press_residuals, **kwargs)
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("Summed $PRESS$ Residual")
        return ax


GridSearchSolution = namedtuple("GridSearchSolution", (
    "tau_sequence", "tau_magnitude", "thresholds", "apexes",
    "target_thresholds"))


class GridPointSolution(object):
    def __init__(self, threshold, lmbda, tau, belongingness_matrix, neighborhood_names, node_names):
        self.threshold = threshold
        self.lmbda = lmbda
        self.tau = tau
        self.belongingness_matrix = belongingness_matrix
        self.neighborhood_names = np.array(neighborhood_names)
        self.node_names = np.array(node_names)

    def __repr__(self):
        return "GridPointSolution(threshold=%0.3f, lmbda=%0.3f, tau=%r)" % (
            self.threshold, self.lmbda, self.tau)

    def clone(self):
        return self.__class__(
            self.threshold,
            self.lmbda,
            self.tau.copy(),
            self.belongingness_matrix.copy(),
            self.neighborhood_names.copy(),
            self.node_names.copy())

    def reindex(self, model):
        node_indices, node_names = self._build_node_index_map(model)
        tau_indices = self._build_neighborhood_index_map(model)

        self.belongingness_matrix = self.belongingness_matrix[node_indices, :][:, tau_indices]
        self.tau = self.tau[tau_indices]

        self.neighborhood_names = model.neighborhood_names
        self.node_names = node_names

    def _build_node_index_map(self, model):
        name_to_new_index = dict()
        name_to_old_index = dict()
        for i, node_name in enumerate(self.node_names):
            name_to_old_index[node_name] = i
            name_to_new_index[node_name] = model.network[node_name].index
        assert len(name_to_new_index) == len(self.node_names)
        ordering = [None for i in range(len(self.node_names))]
        new_name_order = [None for i in range(len(self.node_names))]
        for name, new_index in name_to_new_index.items():
            old_index = name_to_old_index[name]
            ordering[new_index] = old_index
            new_name_order[new_index] = name
        for x in ordering:
            assert x is not None
        return ordering, new_name_order

    def _build_neighborhood_index_map(self, model):
        tau_indices = [model.neighborhood_names.index(name) for name in self.neighborhood_names]
        return tau_indices

    def dump(self, fp):
        fp.write("threshold: %f\n" % (self.threshold,))
        fp.write("lambda: %f\n" % (self.lmbda,))
        fp.write("tau:\n")
        for i, t in enumerate(self.tau):
            fp.write("\t%s\t%f\n" % (self.neighborhood_names[i], t,))
        fp.write("belongingness:\n")
        for g, row in enumerate(self.belongingness_matrix):
            fp.write("\t%s\t" % (self.node_names[g]))
            for i, a_ij in enumerate(row):
                if i != 0:
                    fp.write(",")
                fp.write("%f" % (a_ij,))
            fp.write("\n")
        return fp

    @classmethod
    def load(cls, fp):
        state = "BETWEEN"
        threshold = 0
        lmbda = 0
        tau = []
        belongingness_matrix = []
        neighborhood_names = []
        node_names = []
        for line_number, line in enumerate(fp):
            line = line.strip("\n\r")
            if line.startswith(";"):
                continue
            if line.startswith("threshold:"):
                threshold = float(line.split(":")[1])
                if state in ("TAU", "BELONG"):
                    state = "BETWEEN"
            elif line.startswith("lambda:"):
                lmbda = float(line.split(":")[1])
                if state in ("TAU", "BELONG"):
                    state = "BETWEEN"
            elif line.startswith("tau:"):
                state = "TAU"
            elif line.startswith("belongingness:"):
                state = "BELONG"
            elif line.startswith("\t") or line.startswith("  "):
                if state == "TAU":
                    try:
                        _, name, value = re.split(r"\t|\s{2,}", line)
                    except ValueError as e:
                        print(line_number, line)
                        raise e
                    tau.append(float(value))
                    neighborhood_names.append(name)
                elif state == "BELONG":
                    try:
                        _, name, values = re.split(r"\t|\s{2,}", line)
                    except ValueError as e:
                        print(line_number, line)
                        raise e
                    belongingness_matrix.append([float(t) for t in values.split(",")])
                    node_names.append(name)
                else:
                    state = "BETWEEN"
        return cls(threshold, lmbda, np.array(tau, dtype=np.float64),
                   np.array(belongingness_matrix, dtype=np.float64),
                   neighborhood_names=neighborhood_names,
                   node_names=node_names)


class ThresholdSelectionGridSearch(object):
    def __init__(self, model, network_reduction=None, apex_threshold=0.95, threshold_bias=4.0):
        self.model = model
        self.network_reduction = network_reduction
        self.apex_threshold = apex_threshold
        self.threshold_bias = float(threshold_bias)
        if self.threshold_bias < 1:
            raise ValueError("Threshold Bias must be 1 or greater")

    def has_reduction(self):
        return self.network_reduction is not None and bool(self.network_reduction)

    def explore_grid(self):
        if self.network_reduction is None:
            self.network_reduction = self.model.find_threshold_and_lambda(
                rho=DEFAULT_RHO, threshold_step=0.1, fit_tau=True)
        stack = []
        tau_magnitude = []
        xaxis = []

        for level in self.network_reduction:
            xaxis.append(level.threshold)

        # Pull the distribution slightly to the right
        bias_shift = 1 - (1 / self.threshold_bias)
        # Reduces the influence of the threshold
        bias_scale = self.threshold_bias

        for level in self.network_reduction:
            stack.append(np.array(level.taus).mean(axis=0))
            tau_magnitude.append(
                np.abs(level.optimal_tau).sum() * (
                    (level.threshold / bias_scale) + bias_shift)
            )
        tau_magnitude = np.array(tau_magnitude)
        apex = peak_indices(tau_magnitude)
        xaxis = np.array(xaxis)
        apex = apex[(tau_magnitude[apex] > (tau_magnitude[apex].max() * self.apex_threshold))]
        target_thresholds = [t for t in xaxis[apex]]
        solution = GridSearchSolution(stack, tau_magnitude, xaxis, apex, target_thresholds)
        return solution

    def _get_solution_states(self):
        solution = self.explore_grid()
        states = []
        for i, t in enumerate(solution.target_thresholds):
            states.append(self.network_reduction.searchkey(t))
        return states

    def _get_estimate_for_state(self, state, rho=DEFAULT_RHO, lmbda=None):
        # Get optimal value of lambda based on
        # the PRESS for this group
        if lmbda is None:
            i = np.argmin(state.press_residuals)
            lmbda = state.lambda_values[i]

        # Removes rows from A0
        self.model.set_threshold(state.threshold)

        # tau = self.model.estimate_tau_from_S0(rho, lmbda)
        tau = self.model.estimate_tau_from_S0(rho, lmbda)
        A = self.model.normalized_belongingness_matrix.copy()

        self.model.remove_belongingness_patch()

        return GridPointSolution(state.threshold, lmbda, tau, A,
                                 self.model.neighborhood_names,
                                 self.model.node_names)

    def get_solutions(self, rho=DEFAULT_RHO, lmbda=None):
        states = self._get_solution_states()
        solutions = [self._get_estimate_for_state(
            state, rho=rho, lmbda=lmbda) for state in states]
        self.model.reset()
        return solutions

    def average_solution(self, rho=DEFAULT_RHO, lmbda=None):
        solutions = self.get_solutions(rho=rho, lmbda=lmbda)
        tau_acc = np.zeros_like(solutions[0].tau)
        lmbda_acc = 0
        thresh_acc = 0
        A = np.zeros_like(solutions[0].belongingness_matrix)
        for sol in solutions:
            thresh_acc += sol.threshold
            tau_acc += sol.tau
            lmbda_acc += sol.lmbda
            A += sol.belongingness_matrix
        n = len(solutions)
        thresh_acc /= n
        tau_acc /= n
        lmbda_acc /= n
        A /= n
        # A = ProportionMatrixNormalization.normalize(A, self.model._belongingness_normalization)
        average_solution = GridPointSolution(
            thresh_acc, lmbda_acc, tau_acc, A,
            self.model.neighborhood_names, self.model.node_names)
        return average_solution

    def estimate_phi_observed(self, solution=None, remove_threshold=True, rho=DEFAULT_RHO):
        if solution is None:
            solution = self.average_solution(rho=rho)
        if remove_threshold:
            self.model.reset()
        return self.model.optimize_observed_scores(
            solution.lmbda, solution.belongingness_matrix[self.model.obs_ix, :].dot(solution.tau))

    def estimate_phi_missing(self, solution=None, remove_threshold=True, observed_scores=None):
        if solution is None:
            solution = self.average_solution()
        if remove_threshold:
            self.model.reset()
        if observed_scores is None:
            observed_scores = self.estimate_phi_observed(
                solution=solution, remove_threshold=False)
        t0 = self.model.A0.dot(solution.tau)
        tm = self.model.Am.dot(solution.tau)
        return self.model.compute_missing_scores(observed_scores, t0, tm)

    def annotate_network(self, solution=None, remove_threshold=True, include_missing=True):
        if solution is None:
            solution = self.average_solution()
        if remove_threshold:
            self.model.reset()
        observed_scores = self.estimate_phi_observed(solution, remove_threshold=False)

        if include_missing:
            missing_scores = self.estimate_phi_missing(
                solution, remove_threshold=False,
                observed_scores=observed_scores)

        network = self.model.network.clone()

        for i, ix in enumerate(self.model.obs_ix):
            network[ix].score = observed_scores[i]

        if include_missing:
            for i, ix in enumerate(self.model.miss_ix):
                network[ix].score = missing_scores[i]

        return network

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        solution = self.explore_grid()
        ax.plot(solution.thresholds, solution.tau_magnitude)
        ax.scatter(
            solution.thresholds[solution.apexes],
            solution.tau_magnitude[solution.apexes])
        ax.set_xlabel("Threshold", fontsize=18)
        ax.set_ylabel("Criterion", fontsize=18)
        ax.set_title("Locate Ideal Threshold\nBy Maximizing ${\\bar \\tau_j}$", fontsize=28)
        ax.set_xticks([x_ for i, x_ in enumerate(solution.thresholds) if i % 5 == 0])
        ax.set_xticklabels(["%0.2f" % x_ for i, x_ in enumerate(solution.thresholds) if i % 5 == 0])
        return ax

    def plot_thresholds(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        solution = self.explore_grid()
        ax = self.network_reduction.plot(ax)
        ax.vlines(solution.thresholds[solution.apexes], 0, 1, color='red')
        ax.set_title("Selected Estimation Points for ${\\bar \\tau}$", fontsize=28)
        return ax
