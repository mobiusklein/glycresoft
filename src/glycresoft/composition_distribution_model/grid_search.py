from collections import OrderedDict, namedtuple
import re
from typing import Dict, List, Tuple

import numpy as np

from scipy import sparse

from matplotlib import pyplot as plt

from ms_deisotope.feature_map.profile_transform import peak_indices
from glycresoft.composition_distribution_model.observation import GlycanCompositionSolutionRecord

from glycresoft.task import log_handle
from glycresoft.database.composition_network.graph import CompositionGraph

from .constants import DEFAULT_RHO
from .laplacian_smoothing import LaplacianSmoothingModel


class NetworkReduction(object):
    r"""A mapping-like ordered data structure that maps score thresholds to
    :class:`NetworkTrimmingSolution` instances produced at that score. Used
    for selecting the optimal smoothing factor :math:`\lambda` and during the
    grid search for finding the optimal score threshold.

    Attributes
    ----------
    store : :class:`~.OrderedDict`
        The storage mapping score to solution
    """

    store: OrderedDict
    _invalidated: bool

    def __init__(self, store=None):
        if store is None:
            store = OrderedDict()
        self.store = store
        self._invalidated = True
        self._resort()

    def getkey(self, key):
        """Get the solution whose score threshold matches ``key``

        Parameters
        ----------
        key : float
            The score threshold

        Returns
        -------
        :class:`NetworkTrimmingSolution`
        """
        return self.store[key]

    def getindex(self, ix):
        """Get the ``ix``th solution, regardless of threshold value

        Parameters
        ----------
        ix : int
            The index to return

        Returns
        -------
        :class:`NetworkTrimmingSolution`
        """
        return self.getkey(list(self.store.keys())[ix])

    def searchkey(self, value):
        """Search for the solution whose score threshold
        is closest to ``value``

        Parameters
        ----------
        value : float
            The threshold to search for

        Returns
        -------
        :class:`NetworkTrimmingSolution`
        """
        if self._invalidated:
            self._resort()
        array = list(self.store.keys())
        ix = self.binsearch(array, value)
        key = array[ix]
        return self.getkey(key)

    def put(self, key, value):
        self.store[key] = value
        self._invalidated = True

    def _resort(self):
        self.store = OrderedDict((key, self[key]) for key in sorted(self.store.keys()))
        self._invalidated = False

    def __getitem__(self, key):
        return self.getkey(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __iter__(self):
        return iter(self.store.values())

    def __len__(self):
        return len(self.store)

    @staticmethod
    def binsearch(array, value):
        lo = 0
        hi = len(array)
        i = 0
        while (hi - lo) != 0:
            i = (hi + lo) // 2
            x = array[i]
            if abs(x - value) < 1e-3:
                return i
            elif (hi - lo) == 1:
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
        ax.set_xlabel(r"$S_o$ Threshold", fontsize=18)
        ax.set_ylabel(r"Optimal $\lambda$", fontsize=18)
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
    r"""Hold all the information for the grid search over the smoothing factor
    :math:`\lambda` at score threshold :attr:`threshold`.

    Attributes
    ----------
    lambda_values : :class:`np.ndarray`
        The :math:`\lambda` searched over
    minimum_residuals : float
        The smallest PRESS residuals
    model : :class:`~.LaplacianSmoothingModel`
        The smoothing model used for producing these estimates
    network : :class:`~.CompositionGraph`
        The network structure retained at :attr:`threshold`
    observed : list
        The graph nodes considered observed at this threshold
    optimal_lambda : float
        The best :math:`\lambda` value selected by minimizing the PRESS
    optimal_tau : :class:`np.ndarray`
        The value of :math:`\tau` for the best value of :math:`\lambda`
    press_residuals : :class:`np.ndarray`
        The PRESS at each value of :math:`\lambda`
    taus : list
        The :class:`np.ndarray` for each value of :math:`\lambda`
    threshold : float
        The score threshold used to produce this trimmed network
    updated : bool
        Whether or not this model has changed since the last threshold
    """

    threshold: float
    updated: bool
    observed: List[GlycanCompositionSolutionRecord]

    minimum_residuals: float
    press_residuals: np.ndarray

    optimal_lambda: float
    lambda_values: np.ndarray

    optimal_tau: np.ndarray
    taus: List[np.ndarray]

    model: LaplacianSmoothingModel
    network: CompositionGraph



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

    def copy(self):
        dup = self.__class__(
            self.threshold,
            self.lambda_values.copy(),
            self.press_residuals.copy(),
            self.network.clone(),
            self.observed.copy(),
            list(self.updated),
            list(self.taus),
            self.model)
        return dup

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.lambda_values, self.press_residuals, **kwargs)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"Summed $PRESS$ Residual")
        return ax


GridSearchSolution = namedtuple("GridSearchSolution", (
    "tau_sequence", "tau_magnitude", "thresholds", "apexes",
    "target_thresholds"))



class GridPointTextCodec(object):
    version_code = 1

    def __init__(self, cls_type=None):
        if cls_type is None:
            cls_type = GridPointSolution
        self.cls_type = cls_type

    def load(self, fp):
        state = "BETWEEN"
        threshold = 0
        lmbda = 0
        tau = []
        belongingness_matrix = []
        variance_matrix_tags = {}
        neighborhood_names = []
        node_names = []
        version_code = self.version_code

        for line_number, line in enumerate(fp):
            line = line.strip("\n\r")
            if line.startswith(";version="):
                version_code = int(line.split("=", 1)[1])
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
            elif line.startswith("variance:"):
                state = "VARIANCE"
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
                elif state == 'VARIANCE':
                    try:
                        _, tag, values = re.split(r"\t|\s{2,}", line)
                        values = list(map(float, values.split(",")))
                        variance_matrix_tags[tag] = values
                    except ValueError as e:
                        print(line_number, line)
                        raise e
                else:
                    state = "BETWEEN"
        if variance_matrix_tags:
            vmat_shape = [int(p) for p in variance_matrix_tags['shape']]
            variance_matrix = np.zeros(vmat_shape)
            row_ix = np.array(variance_matrix_tags['rows'], dtype=int)
            col_ix = np.array(variance_matrix_tags['cols'], dtype=int)
            variance_matrix[row_ix, col_ix] = variance_matrix_tags['data']
        else:
            variance_matrix = None

        return self.cls_type(threshold, lmbda, np.array(tau, dtype=np.float64),
                   np.array(belongingness_matrix, dtype=np.float64),
                   neighborhood_names=neighborhood_names,
                   node_names=node_names, variance_matrix=variance_matrix)

    def dump(self, obj, fp):
        fp.write(";version=%s\n" % self.version_code)
        fp.write("threshold: %f\n" % (obj.threshold,))
        fp.write("lambda: %f\n" % (obj.lmbda,))
        fp.write("tau:\n")
        for i, t in enumerate(obj.tau):
            fp.write("\t%s\t%f\n" % (obj.neighborhood_names[i], t,))
        fp.write("belongingness:\n")
        for g, row in enumerate(obj.belongingness_matrix):
            fp.write("\t%s\t" % (obj.node_names[g]))
            for i, a_ij in enumerate(row):
                if i != 0:
                    fp.write(",")
                fp.write("%f" % (a_ij,))
            fp.write("\n")
        fp.write("variance:\n")
        var_coords = sparse.coo_matrix(obj.variance_matrix)
        fp.write('\tshape\t%s\n' % (','.join(map(str, var_coords.shape))))
        fp.write("\trow\t%s\n" % (','.join(map(str, var_coords.row))))
        fp.write("\tcol\t%s\n" % (','.join(map(str, var_coords.col))))
        fp.write("\tdata\t%s\n" % (','.join(map(str, var_coords.data))))
        return fp


class GridPointSolution(object):
    threshold: float
    lmbda: float

    tau: np.ndarray
    belongingness_matrix: np.ndarray
    variance_matrix: np.ndarray

    # Label arrays
    neighborhood_names: np.ndarray
    node_names: np.ndarray

    def __init__(self, threshold, lmbda, tau, belongingness_matrix, neighborhood_names, node_names,
                 variance_matrix=None):
        if variance_matrix is None:
            variance_matrix = np.identity(len(node_names), dtype=np.float64)
        self.threshold = threshold
        self.lmbda = lmbda
        self.tau = np.array(tau, dtype=np.float64)
        self.belongingness_matrix = np.array(belongingness_matrix, dtype=np.float64)
        self.neighborhood_names = np.array(neighborhood_names)
        self.node_names = np.array(node_names)
        self.variance_matrix = np.array(variance_matrix, dtype=np.float64)

    def __eq__(self, other):
        ac = np.allclose
        match = ac(self.threshold, other.threshold) and ac(self.lmbda, other.lmbda) and ac(
            self.tau, other.tau) and ac(self.belongingness_matrix, other.belongingness_matrix) and ac(
            self.variance_matrix, other.variance_matrix)
        if not match:
            return match
        match = np.all(self.node_names == other.node_names) and np.all(
            self.neighborhood_names == other.neighborhood_names)
        return match

    def __ne__(self, other):
        return not (self == other)

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
            self.node_names.copy(),
            self.variance_matrix.copy())

    def reindex(self, model: LaplacianSmoothingModel):
        node_indices, node_names = self._build_node_index_map(model)
        tau_indices = self._build_neighborhood_index_map(model)

        self.belongingness_matrix = self.belongingness_matrix[node_indices, :][:, tau_indices]
        self.tau = self.tau[tau_indices]
        self.variance_matrix = self.variance_matrix[node_indices, node_indices]

        self.neighborhood_names = model.neighborhood_names
        self.node_names = node_names

    def _build_node_index_map(self, model: LaplacianSmoothingModel) -> Tuple[List[int], List[str]]:
        name_to_new_index: Dict[str, int] = dict()
        name_to_old_index: Dict[str, int] = dict()
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

    def _build_neighborhood_index_map(self, model: LaplacianSmoothingModel) -> List[int]:
        tau_indices = [model.neighborhood_names.index(name) for name in self.neighborhood_names]
        return tau_indices

    def dump(self, fp, codec=None):
        if codec is None:
            codec = GridPointTextCodec()
        codec.dump(self, fp)
        return fp

    @classmethod
    def load(cls, fp, codec=None):
        if codec is None:
            codec = GridPointTextCodec(cls)
        inst = codec.load(fp)
        return inst


class NetworkSmoothingModelSolutionBase(object):
    def __init__(self, model):
        self.model = model

    def _get_default_solution(self, *args, **kwargs):
        raise NotImplementedError()

    def estimate_phi_observed(self, solution=None, remove_threshold=True, rho=DEFAULT_RHO):
        if solution is None:
            solution = self._get_default_solution(rho=rho)
        if remove_threshold:
            self.model.reset()
        return self.model.optimize_observed_scores(
            solution.lmbda, solution.belongingness_matrix[self.model.obs_ix, :].dot(solution.tau))

    def estimate_phi_missing(self, solution=None, remove_threshold=True, observed_scores=None):
        if solution is None:
            solution = self._get_default_solution()
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
            solution = self._get_default_solution()
        if remove_threshold:
            self.model.reset()
        observed_scores = self.estimate_phi_observed(solution, remove_threshold=False)

        if include_missing:
            missing_scores = self.estimate_phi_missing(
                solution, remove_threshold=False,
                observed_scores=observed_scores)

        network = self.model.network.clone()
        network.neighborhoods = self.model.neighborhood_walker.neighborhoods.clone()
        for i, ix in enumerate(self.model.obs_ix):
            network[ix].score = observed_scores[i]

        if include_missing:
            for i, ix in enumerate(self.model.miss_ix):
                network[ix].score = missing_scores[i]
                network[ix].marked = True

        return network


class ThresholdSelectionGridSearch(NetworkSmoothingModelSolutionBase):
    def __init__(self, model, network_reduction=None, apex_threshold=0.95, threshold_bias=4.0):
        super(ThresholdSelectionGridSearch, self).__init__(model)
        self.network_reduction = network_reduction
        self.apex_threshold = apex_threshold
        self.threshold_bias = float(threshold_bias)
        if self.threshold_bias < 1:
            raise ValueError("Threshold Bias must be 1 or greater")

    def _get_default_solution(self, *args, **kwargs):
        return self.average_solution(*args, **kwargs)

    def has_reduction(self):
        return self.network_reduction is not None and bool(self.network_reduction)

    def explore_grid(self):
        if self.network_reduction is None:
            self.network_reduction = self.model.find_threshold_and_lambda(
                rho=DEFAULT_RHO, threshold_step=0.1, fit_tau=True)
        log_handle.log("... Exploring Grid Landscape")
        stack = []
        tau_magnitude = []
        thresholds = []

        for level in self.network_reduction:
            thresholds.append(level.threshold)

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
        if len(tau_magnitude) == 0:
            # No solutions, so these will be empty
            return GridSearchSolution(stack, tau_magnitude, thresholds, np.array([]), thresholds)
        elif len(tau_magnitude) <= 2:
            apex = np.argmax(tau_magnitude)
        elif len(tau_magnitude) > 2:
            apex = peak_indices(tau_magnitude)
            if len(apex) == 0:
                apex = np.array([np.argmax(tau_magnitude)])

        thresholds = np.array(thresholds)
        apex_threshold = tau_magnitude[apex].max() * self.apex_threshold
        if apex_threshold != 0:
            apex = apex[(tau_magnitude[apex] > apex_threshold)]
        else:
            # The tau threshold may be 0, in which case any point will do, but this
            # solution carries no generalization.
            apex = apex[(tau_magnitude[apex] >= apex_threshold)]
        target_thresholds = [t for t in thresholds[apex]]
        solution = GridSearchSolution(stack, tau_magnitude, thresholds, apex, target_thresholds)
        log_handle.log("... %d Candidate Solutions" % (len(target_thresholds),))
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

        tau = self.model.estimate_tau_from_S0(rho, lmbda)
        A = self.model.normalized_belongingness_matrix.copy()

        # Restore removed rows from A0
        self.model.remove_belongingness_patch()

        variance_matrix = np.identity(len(self.model.network)) * 0
        variance_matrix[self.model.obs_ix, self.model.obs_ix] = np.diag(self.model.variance_matrix)

        return GridPointSolution(state.threshold, lmbda, tau, A,
                                 self.model.neighborhood_names,
                                 self.model.node_names,
                                 variance_matrix=variance_matrix)

    def get_solutions(self, rho=DEFAULT_RHO, lmbda=None):
        states = self._get_solution_states()
        solutions = [self._get_estimate_for_state(
            state, rho=rho, lmbda=lmbda) for state in states]
        self.model.reset()
        return solutions

    def average_solution(self, rho=DEFAULT_RHO, lmbda=None):
        solutions = self.get_solutions(rho=rho, lmbda=lmbda)
        if len(solutions) == 0:
            return None
        tau_acc = np.zeros_like(solutions[0].tau)
        lmbda_acc = 0
        thresh_acc = 0
        # variance_matrix = np.zeros_like()
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
        variance_matrix = np.identity(len(self.model.network)) * 0
        variance_matrix[self.model.obs_ix, self.model.obs_ix] = np.diag(self.model.variance_matrix)
        average_solution = GridPointSolution(
            thresh_acc, lmbda_acc, tau_acc, A,
            self.model.neighborhood_names, self.model.node_names,
            variance_matrix=variance_matrix)
        return average_solution

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
        ax.set_xticklabels(["%0.2f" % x_ for i, x_ in enumerate(solution.thresholds) if i % 5 == 0], rotation=90)
        return ax

    def plot_thresholds(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        solution = self.explore_grid()
        ax = self.network_reduction.plot(ax)
        ax.vlines(solution.thresholds[solution.apexes], 0, 1, color='red')
        ax.set_title("Selected Estimation Points for ${\\bar \\tau}$", fontsize=28)
        return ax

    def fit(self, rho=DEFAULT_RHO, lmbda=None):
        solution = self.average_solution(rho=rho, lmbda=lmbda)
        if solution is None:
            raise ValueError("Could not fit model. No acceptable solution found.")
        return NetworkSmoothingModelFit(self.model, solution)


class NetworkSmoothingModelFit(NetworkSmoothingModelSolutionBase):
    def __init__(self, model, solution):
        super(NetworkSmoothingModelFit, self).__init__(model)
        self.solution = solution

    def _get_default_solution(self, *args, **kwargs):
        return self.solution

    def __repr__(self):
        return "{self.__class__.__name__}({self.model}, {self.solution})".format(self=self)
