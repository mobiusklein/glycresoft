from collections import OrderedDict, defaultdict

import numpy as np
from scipy import linalg
import scipy.linalg
from matplotlib import pyplot as plt

from glycan_profiling.task import TaskBase
from glycan_profiling.database import composition_network
from glycan_profiling.scoring import network_scoring


class PriorBuilder(object):

    def __init__(self, network):
        self.network = network
        self.value = np.zeros(len(network))

    def bias(self, target, v):
        ix = self.network[target].index
        self.value[ix] += v
        return self

    def common(self, v):
        self.value += v
        return self

    def __getitem__(self, i):
        ix = self.network[i].index
        return self.value[ix]


class CompositionDistributionFit(object):

    def __init__(self, pi, network):
        self.pi = pi
        self.network = network

    def index_of(self, composition):
        return self.network[composition].index

    def __getitem__(self, composition):
        return self.pi[self.index_of(composition)]

    def __iter__(self):
        for i in range(len(self)):
            yield self.network[i], self.pi[i]

    def __len__(self):
        return len(self.network)

    def __repr__(self):
        return "CompositionDistributionFit()"


class CompositionDistributionUpdater(TaskBase):
    etol = 1e-6

    def __init__(self, network, solutions, prior=0.01, common_weight=1., precision_scale=1.):
        self.network = network
        self.solutions = solutions
        self.common_weight = common_weight
        self.precision_scale = precision_scale
        if isinstance(prior, (int, float)):
            prior = np.ones(len(network)) * prior
        self.prior = prior * precision_scale + common_weight

        # Dirichlet Mean
        self.pi0 = self.prior / self.prior.sum()

        self.sol_map = {
            sol.composition: sol.score for sol in solutions if sol.composition is not None}

        self.observations = np.array(
            [self.sol_map.get(node.composition.serialize(), 0) for node in network.nodes])
        self.iterations = 0

    def index_of(self, composition):
        return self.network[composition].index

    def convergence(self, pi_next, pi_last):
        d = np.abs(pi_last).sum()
        v = (np.abs(pi_next - pi_last).sum()) / d
        return v

    def update_pi(self, pi_array):
        pi2 = np.zeros_like(pi_array)
        total_score_pi = (self.observations * pi_array).sum()
        total_w = ((self.prior - 1).sum() + 1)
        for i in range(len(self.prior)):
            pi2[i] = ((self.prior[i] - 1) + (self.observations[i] *
                                             pi_array[i]) / total_score_pi) / total_w
            assert pi2[i] >= 0, (self.prior[i], pi_array[i])
        return pi2

    def optimize_pi(self, pi, maxiter=100, **kwargs):
        pi_last = pi
        pi_next = self.update_pi(pi_last)

        self.iterations = 0
        converging = float('inf')
        while converging > self.etol and self.iterations < maxiter:
            if self.iterations % 100 == 0:
                self.log("%f, %d" % (converging, self.iterations))
            pi_last = pi_next
            pi_next = self.update_pi(pi_last)
            self.iterations += 1
            converging = self.convergence(pi_next, pi_last)
        if converging < self.etol:
            self.log("Converged in %d iterations" % self.iterations)
        else:
            self.log("Failed to converge in %d iterations (%f)" %
                     (self.iterations, converging))

        return pi_next

    def fit(self, **kwargs):
        return CompositionDistributionFit(
            self.optimize_pi(self.pi0, **kwargs), self.network)


def adjacency_matrix(network):
    A = np.zeros((len(network), len(network)))
    for edge in network.edges:
        i, j = edge.node1.index, edge.node2.index
        A[i, j] = 1
        A[j, i] = 1
    for i in range(A.shape[0]):
        A[i, i] = 0
    return A


def weighted_adjacency_matrix(network):
    A = np.zeros((len(network), len(network)))
    A[:] = 1. / float('inf')
    for edge in network.edges:
        i, j = edge.node1.index, edge.node2.index
        A[i, j] = 1. / edge.order
        A[j, i] = 1. / edge.order
    for i in range(A.shape[0]):
        A[i, i] = 0
    return A


def degree_matrix(network):
    degrees = [len(n.edges) for n in network]
    return np.diag(degrees)


def weighted_degree_matrix(network):
    degrees = [sum(1. / e.order for e in n.edges) for n in network]
    return np.diag(degrees)


def laplacian_matrix(network):
    return degree_matrix(network) - adjacency_matrix(network)


def weighted_laplacian_matrix(network):
    return weighted_degree_matrix(network) - weighted_adjacency_matrix(network)


# def assign_network(network, observed):
#     rns = network_scoring.RenormalizingNetworkScorerDifferenceMethod(
#         observed, network.clone())
#     rns.build_solution_map()
#     rns.assign_network()
#     return rns.network


def assign_network(network, observed, copy=True):
    if copy:
        network = network.clone()
    solution_map = {}
    for case in observed:
        if case.glycan_composition is None:
            continue
        s = solution_map.get(case.glycan_composition)
        if s is None or s.score < case.score:
            solution_map[case.glycan_composition] = case

    for node in network.nodes:
        node.internal_score = 0
        node._temp_score = 0

    for composition, solution in solution_map.items():
        try:
            node = network[composition]
            node._temp_score = node.internal_score = solution.internal_score
        except KeyError:
            # Not all exact compositions have nodes
            continue
    return network


def network_indices(network, threshold=0.0001):
    missing = []
    observed = []
    for node in network:
        if node._temp_score < threshold:
            missing.append(node.index)
        else:
            observed.append(node.index)
    return observed, missing


def make_blocks(network, observed, threshold=0.0001):
    network = assign_network(network, observed)
    structure_matrix = weighted_laplacian_matrix(network)
    observed_indices, missing_indices = network_indices(network, threshold)
    oo_block = structure_matrix[observed_indices, :][:, observed_indices]
    om_block = structure_matrix[observed_indices, :][:, missing_indices]
    mo_block = structure_matrix[missing_indices, :][:, observed_indices]
    mm_block = structure_matrix[missing_indices, :][:, missing_indices]
    return {"oo": oo_block, "om": om_block, "mo": mo_block, "mm": mm_block}


def _make_blocks(network, threshold=0.0001):
    structure_matrix = weighted_laplacian_matrix(network)
    observed_indices, missing_indices = network_indices(network, threshold)
    oo_block = structure_matrix[observed_indices, :][:, observed_indices]
    om_block = structure_matrix[observed_indices, :][:, missing_indices]
    mo_block = structure_matrix[missing_indices, :][:, observed_indices]
    mm_block = structure_matrix[missing_indices, :][:, missing_indices]
    return {"oo": oo_block, "om": om_block, "mo": mo_block, "mm": mm_block}


def compute_missing_scores(blocks, observed_scores, t0=0., tm=0.):
    return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm


def optimize_observed_scores(blocks, lmbda, observed_scores, t0=0.):
    S = lmbda * (blocks["oo"] - blocks["om"].dot(linalg.inv(
                 blocks['mm'])).dot(blocks["mo"]))
    B = np.eye(len(observed_scores)) + S
    return linalg.inv(B).dot(observed_scores - t0) + t0


def smooth_network(network, observations, lmbda=0.2, t0=0., tm=0., threshold=0.4, fit_t=False):
    network = assign_network(network, observations)
    structure_matrix = weighted_laplacian_matrix(network)

    observed_indices, missing_indices = network_indices(network, threshold)
    oo_block = structure_matrix[observed_indices, :][:, observed_indices]
    om_block = structure_matrix[observed_indices, :][:, missing_indices]
    mo_block = structure_matrix[missing_indices, :][:, observed_indices]
    mm_block = structure_matrix[missing_indices, :][:, missing_indices]
    blocks = {"oo": oo_block, "om": om_block, "mo": mo_block, "mm": mm_block}

    observed_scores = np.array([
        node.score for node in network[observed_indices]
    ])
    observed_labels = [
        node for node in network[observed_indices]
    ]
    # missing_scores = compute_missing_scores(blocks, observed_scores, t0, tm)
    update_obs_scores = optimize_observed_scores(
        blocks, lmbda, observed_scores, t0)
    update_missing_scores = compute_missing_scores(
        blocks, update_obs_scores, t0, tm)

    if fit_t:
        for node, score in zip(observed_labels, update_obs_scores):
            node.score = score
        nhw = composition_network.NeighborhoodWalker(network)
        nhann = composition_network.DistanceWeightedNeighborhoodAnalyzer(nhw)
        t0_update = []
        for node in observed_labels:
            t0_update.append(nhann[node])
        update_obs_scores = optimize_observed_scores(
            blocks, lmbda, observed_scores, t0_update)
        update_missing_scores = compute_missing_scores(
            blocks, update_obs_scores, t0_update, tm)

    return update_obs_scores, update_missing_scores, observed_labels, observed_scores


class GlycomeModel(object):

    def __init__(self, observed_compositions, network, belongingness_matrix=None):
        self._network = network
        self.network = assign_network(
            network.clone(), observed_compositions)
        self._observed_compositions = observed_compositions

        self.neighborhood_walker = composition_network.NeighborhoodWalker(
            self.network)
        self.neighborhood_analyzer = composition_network.DistanceWeightedNeighborhoodAnalyzer(
            self.neighborhood_walker)
        self.obs_ix, self.miss_ix = network_indices(self.network)

        block_L = self.block_L = make_blocks(
            self.network.clone(), self._observed_compositions)
        L_mm_inv = np.linalg.inv(block_L['mm'])
        self.L_oo_inv = np.linalg.pinv(
            block_L["oo"] - (block_L['om'].dot(L_mm_inv).dot(block_L['mo'])))

        # Expensive Step
        if belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()
        else:
            self.belongingness_matrix = belongingness_matrix

        # Initialize Names
        self.normalized_belongingness_matrix = None
        self.A0 = None
        self._belongingness_normalization = None
        # Normalize and populate
        self.normalize_belongingness('colrow')
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]
        self.S0 = np.array([g.score for g in self.network[self.obs_ix]])
        self.C0 = ([g for g in self.network[self.obs_ix]])

    def set_threshold(self, threshold):
        accepted = [
            g for g in self._observed_compositions if g.score > threshold]
        self.network = assign_network(self._network.clone(), accepted)
        self.neighborhood_walker = composition_network.NeighborhoodWalker(self.network)
        self.neighborhood_analyzer = composition_network.DistanceWeightedNeighborhoodAnalyzer(
            self.neighborhood_walker)
        # self.obs_ix, self.miss_ix = self.neighborhood_analyzer.network_indices()
        self.obs_ix, self.miss_ix = network_indices(self.network)
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]
        self.S0 = np.array([g.score for g in self.network[self.obs_ix]])
        self.C0 = ([g for g in self.network[self.obs_ix]])

        block_L = self.block_L = make_blocks(
            self.network.clone(), accepted)
        L_mm_inv = np.linalg.inv(block_L['mm'])
        self.L_oo_inv = np.linalg.pinv(
            block_L["oo"] - (block_L['om'].dot(L_mm_inv).dot(block_L['mo'])))

    def normalize_belongingness(self, method='col'):
        if method == 'col':
            self.normalized_belongingness_matrix = (
                self.belongingness_matrix / self.belongingness_matrix.sum(axis=0))
        elif method == 'row':
            self.normalized_belongingness_matrix = (
                self.belongingness_matrix / self.belongingness_matrix.sum(axis=1).reshape((-1, 1)))
        elif method == 'colrow':
            self.normalized_belongingness_matrix = (
                self.belongingness_matrix / self.belongingness_matrix.sum(axis=0))
            self.normalized_belongingness_matrix = (
                self.normalized_belongingness_matrix / self.normalized_belongingness_matrix.sum(
                    axis=1).reshape((-1, 1)))
        elif method == 'none' or method is None:
            self.normalized_belongingness_matrix = self.belongingness_matrix
        else:
            raise ValueError(method)
        self._belongingness_normalization = method
        self.normalized_belongingness_matrix[np.isnan(self.normalized_belongingness_matrix)] = 0.0
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]

    def build_belongingness_matrix(self):
        neighborhood_count = len(self.neighborhood_walker.neighborhoods)
        belongingness_matrix = np.zeros(
            (len(self.network), neighborhood_count))

        for node in self.network:
            was_in = self.neighborhood_walker.neighborhood_assignments[node]
            for i, neighborhood in enumerate(self.neighborhood_walker.neighborhoods):
                if neighborhood.name in was_in:
                    belongingness_matrix[node.index, i] = self.neighborhood_walker.compute_belongingness(
                        node, neighborhood.name)
        return belongingness_matrix

    def estimate_tau_from_S0(self, rho, lmda, sigma2=1.0):
        X = ((rho / sigma2) * np.eye(len(self.S0))) + (
            (1. / (lmda * sigma2)) * self.L_oo_inv) + self.A0.dot(self.A0.T)
        X = np.linalg.pinv(X)
        return self.A0.T.dot(X).dot(self.S0)

    def sample_tau(self, rho, lmda):
        sigma_est = np.std(self.S0)
        mu_tau = self.estimate_tau_from_S0(rho, lmda)
        return np.random.multivariate_normal(mu_tau, np.eye(len(mu_tau)).dot(sigma_est ** 2))

    def phi_given_tau(self, tau, lmda):
        return np.random.multivariate_normal(self.A0.dot(tau), (1. / lmda) * self.L_oo_inv)

    def optimize_observed_scores(self, lmda, t0=0):
        blocks = self.block_L
        L = lmda * (blocks["oo"] - blocks["om"].dot(np.linalg.inv(
            blocks['mm'])).dot(blocks["mo"]))
        B = np.eye(len(self.S0)) + L
        return np.linalg.inv(B).dot(self.S0 - t0) + t0

    def compute_missing_scores(self, observed_scores, t0=0., tm=0.):
        blocks = self.block_L
        return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm

    def find_optimal_lambda(self, lambda_max=1, step=0.01, threshold=0.0001):
        obs = []
        missed = []
        network = self.network.clone()
        for node in network:
            if node.score < threshold:
                missed.append(node)
            else:
                obs.append(node.score)
        lambda_values = np.arange(0.01, lambda_max, step)
        ident = np.eye(len(obs))
        press = []
        for node in missed:
            network.remove_node(node, limit=5)
        wpl = weighted_laplacian_matrix(network)

        for lambd in lambda_values:
            A = ident + lambd * wpl
            C = scipy.linalg.cholesky(A)
            # The solution for theta, the updated scores
            T = scipy.linalg.cho_solve((C, False), obs)
            H = np.linalg.inv(A)
            press_value = sum(
                ((obs - T) / (1 - (np.diag(H) - np.finfo(float).eps))) ** 2) / len(obs)
            press.append(press_value)
        return lambda_values, np.array(press)

    def find_threshold_and_lambda(self, lambda_max=1., lambda_step=0.01, threshold_start=0.,
                                  threshold_step=0.2):
        solutions = NetworkReduction()
        limit = max(self.S0)
        start = max(min(self.S0), threshold_start)
        current_network = self.network.clone()
        for threshold in np.arange(start, limit, threshold_step):
            obs = []
            missed = []
            network = current_network.clone()
            for node in network:
                if node.score < threshold:
                    missed.append(node)
                else:
                    obs.append(node.score)
            if len(obs) == 0:
                break
            lambda_values = np.arange(0.01, lambda_max, lambda_step)
            ident = np.eye(len(obs))
            press = []
            for node in missed:
                network.remove_node(node, limit=5)
            wpl = weighted_laplacian_matrix(network)

            for lambd in lambda_values:
                A = ident + lambd * wpl
                C = scipy.linalg.cholesky(A)
                # The solution for theta, the updated scores
                T = scipy.linalg.cho_solve((C, False), obs)
                H = np.linalg.inv(A)
                press_value = sum(
                    ((obs - T) / (1 - (np.diag(H) - np.finfo(float).eps))) ** 2) / len(obs)
                press.append(press_value)
            solutions[threshold] = NetworkTrimmingSearchSolution(
                threshold, lambda_values, np.array(press), len(network))
            current_network = network
        return solutions


class GroupBelongingnessMatrix(object):

    @classmethod
    def from_model(cls, model):
        mat = model.normalized_belongingness_matrix
        groups = model.neighborhood_walker.neighborhood_names()
        members = model.network.nodes
        return cls(mat, groups, members)

    def __init__(self, belongingness_matrix, groups, members):
        self.belongingness_matrix = belongingness_matrix
        self.groups = [str(x) for x in groups]
        self.members = [str(x) for x in members]
        self._column_indices = OrderedDict([(k, i) for i, k in enumerate(groups)])
        self._member_indices = OrderedDict([(k, i) for i, k in enumerate(self.members)])

    def _column_indices_by_name(self, names):
        if isinstance(names, basestring):
            names = [names]
        indices = [self._column_indices[n] for n in names]
        return indices

    def _member_indices_by_name(self, names):
        if isinstance(names, (basestring, composition_network.CompositionGraphNode,
                              composition_network.GlycanComposition,
                              composition_network.GlycanCompositionProxy)):
            names = [names]
        indices = [self._member_indices[n] for n in names]
        return indices

    def _coerce_member(self, names):
        if isinstance(names, (basestring, composition_network.CompositionGraphNode,
                              composition_network.GlycanComposition,
                              composition_network.GlycanCompositionProxy)):
            names = [names]
        return names

    def _coerce_column(self, names):
        if isinstance(names, basestring):
            names = [names]
        return names

    def get(self, rows=None, cols=None):
        matrix = self.belongingness_matrix
        if rows is not None:
            rows = self._coerce_member(rows)
            row_ix = self._member_indices_by_name(rows)
            matrix = matrix[row_ix, :]
        else:
            rows = self.members
        if cols is not None:
            cols = self._coerce_column(cols)
            col_ix = self._column_indices_by_name(cols)
            matrix = matrix[:, col_ix]
        else:
            cols = self.groups
        return self.__class__(matrix, cols, rows)

    def __getitem__(self, ij):
        return self.belongingness_matrix[ij]

    def __repr__(self):
        column_label_width = max(map(len, self.groups))
        row_label_width = max(map(len, self.members))
        rows = []
        top_row = [' ' * row_label_width]
        for col in self.groups:
            top_row.append(col.center(column_label_width))
        rows.append('|'.join(top_row))
        for i, member in enumerate(self.members):
            row = [member.ljust(row_label_width)]
            vals = self.belongingness_matrix[i, :]
            for val in vals:
                row.append(("%0.3f" % val).center(column_label_width))
            rows.append("|".join(row))
        return '\n'.join(rows)


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
        key = array[ix + 1]
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

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        x = self.store.keys()
        y = [v.opt_lambda for v in self.store.values()]
        ax.plot(x, y)
        xbound = ax.get_xlim()
        ybound = ax.get_ylim()
        ax.scatter(x, y)
        ax.set_xlim(*xbound)
        ax.set_ylim(*ybound)
        ax.set_xlabel("$S_o$ Threshold")
        ax.set_ylabel("Optimal $\lambda$")
        return ax

    def minimum_threshold_for_lambda(self, lmbda_target):
        best = None
        for value in reversed(self.store.values()):
            if best is None:
                best = value
                continue
            if abs(best.opt_lambda - lmbda_target) >= abs(value.opt_lambda - lmbda_target):
                if value.threshold < best.threshold:
                    best = value
        return best

    def press_weighted_mean_threshold(self):
        vals = list(self)
        return np.average(np.array(
            [v.threshold for v in vals]), weights=np.array(
            [v.press_residuals.min() for v in vals]))


class NetworkTrimmingSearchSolution(object):

    def __init__(self, threshold, lambda_values, press_residuals, n_kept):
        self.threshold = threshold
        self.lambda_values = lambda_values
        self.press_residuals = press_residuals
        self.n_kept = n_kept
        self.optimal_lambda = self.lambda_values[np.argmin(self.press_residuals)]
        self.minimum_residuals = self.press_residuals.min()

    @property
    def opt_lambda(self):
        return self.optimal_lambda

    def __repr__(self):
        min_press = min(self.press_residuals)
        opt_lambda = self.opt_lambda
        return "NetworkTrimmingSearchSolution(%f, %d, %0.3f -> %0.3e)" % (
            self.threshold, self.n_kept, opt_lambda, min_press)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.lambda_values, self.press_residuals)
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("Summed $PRESS$ Residual")
        return ax
