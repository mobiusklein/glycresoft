from collections import OrderedDict, namedtuple, defaultdict

from six import string_types as basestring

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

from glycan_profiling.task import TaskBase
from glycan_profiling.database.composition_network import (
    NeighborhoodWalker, CompositionGraphNode,
    GlycanComposition, GlycanCompositionProxy)

from ms_deisotope.feature_map.profile_transform import peak_indices


DEFAULT_LAPLACIAN_REGULARIZATION = 1.0
RESET_THRESHOLD_VALUE = 1e-3


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
            self.iterations += 1
            if self.iterations % 100 == 0:
                self.log("%f, %d" % (converging, self.iterations))
            pi_last = pi_next
            pi_next = self.update_pi(pi_last)
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


class BlockLaplacian(object):
    def __init__(self, network, threshold=0.0001, regularize=1.0):
        structure_matrix = weighted_laplacian_matrix(network)
        structure_matrix = structure_matrix + (np.eye(structure_matrix.shape[0]) * regularize)
        observed_indices, missing_indices = network_indices(network, threshold)

        oo_block = structure_matrix[observed_indices, :][:, observed_indices]
        om_block = structure_matrix[observed_indices, :][:, missing_indices]
        mo_block = structure_matrix[missing_indices, :][:, observed_indices]
        mm_block = structure_matrix[missing_indices, :][:, missing_indices]

        self.matrix = structure_matrix
        self.blocks = {"oo": oo_block, "om": om_block, "mo": mo_block, "mm": mm_block}

        self.obs_ix = observed_indices
        self.miss_ix = missing_indices

        self.L_mm_inv = np.linalg.inv(self['mm'])
        self.L_oo_inv = np.linalg.pinv(
            self["oo"] - (self['om'].dot(self.L_mm_inv).dot(self['mo'])))

        self.regularize = regularize
        self.threshold = threshold

    def __getitem__(self, k):
        return self.blocks[k]

    def __repr__(self):
        return "BlockLaplacian(%s)" % (', '.join({
            "%s: %r" % (k, v.shape) for k, v in self.blocks.items()
        }))


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
        if node.internal_score < threshold:
            missing.append(node.index)
        else:
            observed.append(node.index)
    return observed, missing


def make_blocks(network, observed, threshold=0.0001, regularize=DEFAULT_LAPLACIAN_REGULARIZATION):
    network = assign_network(network, observed)
    return BlockLaplacian(network, threshold, regularize)


def _reference_compute_missing_scores(blocks, observed_scores, t0=0., tm=0.):
    return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm


def _reference_optimize_observed_scores(blocks, lmbda, observed_scores, t0=0.):
    S = lmbda * (blocks["oo"] - blocks["om"].dot(linalg.inv(
                 blocks['mm'])).dot(blocks["mo"]))
    B = np.eye(len(observed_scores)) + S
    return linalg.inv(B).dot(observed_scores - t0) + t0


def scale_network(network, maximum):
    relmax = max([n.score for n in network.nodes])
    for node in network.nodes:
        node.score = node.score / relmax * maximum
    return network


class LaplacianSmoothingModel(object):
    def __init__(self, network, belongingness_matrix, threshold,
                 regularize=DEFAULT_LAPLACIAN_REGULARIZATION, neighborhood_walker=None,
                 belongingness_normalization='colrow'):
        self.network = network

        if neighborhood_walker is None:
            self.neighborhood_walker = NeighborhoodWalker(self.network)
        else:
            self.neighborhood_walker = neighborhood_walker

        self.belongingness_matrix = belongingness_matrix
        self.threshold = threshold

        self.obs_ix, self.miss_ix = network_indices(self.network, self.threshold)
        self.block_L = BlockLaplacian(
            self.network, threshold, regularize=regularize)

        self.S0 = [node.score for node in self.network[self.obs_ix]]
        self.A0 = belongingness_matrix[self.obs_ix, :]
        self.Am = belongingness_matrix[self.miss_ix, :]
        self.C0 = [node for node in self.network[self.obs_ix]]

        # Intentionally use column-row normalization during fitting?
        self._belongingness_normalization = belongingness_normalization

    @property
    def L_mm_inv(self):
        return self.block_L.L_mm_inv

    @property
    def L_oo_inv(self):
        return self.block_L.L_oo_inv

    def optimize_observed_scores(self, lmda, t0=0):
        blocks = self.block_L
        L = lmda * (blocks["oo"] - blocks["om"].dot(self.L_mm_inv).dot(blocks["mo"]))
        B = np.eye(len(self.S0)) + L
        return np.linalg.inv(B).dot(self.S0 - t0) + t0

    def compute_missing_scores(self, observed_scores, t0=0., tm=0.):
        blocks = self.block_L
        return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm

    def compute_projection_matrix(self, lmbda):
        A = np.eye(self.L_oo_inv.shape[0]) + self.L_oo_inv * (1. / lmbda)
        H = np.linalg.pinv(A)
        return H

    def compute_press(self, observed, updated, projection_matrix):
        press = np.sum(((observed - updated) / (
            1 - (np.diag(projection_matrix) - np.finfo(float).eps))) ** 2) / len(observed)
        return press

    def estimate_tau_from_S0(self, rho, lmda, sigma2=1.0):
        X = ((rho / sigma2) * np.eye(len(self.S0))) + (
            (1. / (lmda * sigma2)) * self.L_oo_inv) + self.A0.dot(self.A0.T)
        X = np.linalg.pinv(X)
        return self.A0.T.dot(X).dot(self.S0)

    def get_belongingness_patch(self):
        updated_belongingness = BelongingnessMatrixPatcher.patch(self)
        updated_belongingness = ProportionMatrixNormalization.normalize(
            updated_belongingness, self._belongingness_normalization)
        return updated_belongingness

    def apply_belongingness_patch(self):
        updated_belongingness = self.get_belongingness_patch()
        self.belongingness_matrix = updated_belongingness
        self.A0 = self.belongingness_matrix[self.obs_ix, :]


class ProportionMatrixNormalization(object):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def normalize_columns(self):
        self.matrix = self.matrix / self.matrix.sum(axis=0)

    def normalize_rows(self):
        self.matrix = self.matrix / self.matrix.sum(axis=1).reshape((-1, 1))

    def normalize_columns_and_rows(self):
        self.normalize_columns()
        self.normalize_rows()

    def normalize_columns_scaled(self, scaler=2.0):
        self.normalize_columns()
        self.matrix = self.matrix * scaler

    def clean(self):
        self.matrix[np.isnan(self.matrix)] = 0.0

    def __array__(self):
        return self.matrix

    @classmethod
    def normalize(cls, matrix, method='colrow'):
        self = cls(matrix)
        if method == 'col':
            self.normalize_columns()
        elif method == 'row':
            self.normalize_rows()
        elif method == 'colrow':
            self.normalize_columns_and_rows()
        elif method.startswith("col") and method[3:4].isdigit():
            scale = float(method[3:])
            self.normalize_columns_scaled(scale)
        elif method == 'none' or method is None:
            pass
        else:
            raise ValueError("Unknown Normalization Method %r" % method)
        self.clean()
        return self.matrix


def _has_glycan_composition(x):
    try:
        gc = x.glycan_composition
        return gc is not None
    except AttributeError:
        return False


class VariableObservationAggregation(object):
    def __init__(self, observations):
        self.aggregation = defaultdict(list)
        self.observations = observations
        self.collect()

    def collect(self):
        for obs in self.observations:
            self.aggregation[obs.glycan_composition].append(obs)

    def estimate_summaries(self):
        means = OrderedDict()
        variances = OrderedDict()
        for key, values in self.aggregation.items():
            means[key] = np.mean([v.score for v in values])
            variances[key] = 1. / len(values)
        return means, variances

    def update(self, network):
        means, variances = self.estimate_summaries()
        n = len(network)
        observed_scores = np.zeros(n)
        variance_matrix = np.eye(n)
        nodes = []
        for key, value in observed_scores.items():
            node = network[key]
            observed_scores[node.index] = value
            node.score = value
            nodes.append(node)
            variance_matrix[node.index, node.index] = variances[key]
        observed_scores = observed_scores[observed_scores > 0]
        nodes.sort(key=lambda x: x.index)
        return observed_scores, variance_matrix, nodes

    @classmethod
    def extract(cls, observations, network):
        inst = cls(observations)
        return inst.update(network)

    @classmethod
    def from_model(cls, model):
        observations = model.observations
        network = model.network
        return cls.extract(observations, network)


class GlycomeModel(LaplacianSmoothingModel):

    def __init__(self, observed_compositions, network, belongingness_matrix=None,
                 regularize=DEFAULT_LAPLACIAN_REGULARIZATION,
                 belongingness_normalization='colrow'):
        observed_compositions = [
            o for o in observed_compositions if _has_glycan_composition(o)]
        self._network = network
        self._observed_compositions = observed_compositions

        self.network = assign_network(network.clone(), observed_compositions)

        self.neighborhood_walker = NeighborhoodWalker(
            self.network)

        self.neighborhood_names = self.neighborhood_walker.neighborhood_names()

        self.obs_ix, self.miss_ix = network_indices(self.network)

        self.block_L = BlockLaplacian(self.network, regularize=regularize)
        self.threshold = self.block_L.threshold

        # Initialize Names
        self.normalized_belongingness_matrix = None
        self.A0 = None
        self._belongingness_normalization = None
        self.S0 = []
        self.C0 = []

        # Expensive Step
        if belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()
        else:
            self.belongingness_matrix = np.array(belongingness_matrix)

        # Normalize and populate
        self.normalize_belongingness(belongingness_normalization)
        self._populate()

    def _populate(self):
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]
        self.Am = self.normalized_belongingness_matrix[self.miss_ix, :]
        self.S0 = np.array([g.score for g in self.network[self.obs_ix]])
        self.C0 = ([g for g in self.network[self.obs_ix]])

    def set_threshold(self, threshold):
        accepted = [
            g for g in self._observed_compositions if g.score > threshold]
        if len(accepted) == 0:
            raise ValueError("Threshold %f produces an empty observed set" % (threshold,))
        self.network = assign_network(self._network.clone(), accepted)

        self.obs_ix, self.miss_ix = network_indices(self.network)
        self._populate()

        self.block_L = BlockLaplacian(self.network, threshold=threshold, regularize=self.block_L.regularize)
        self.threshold = self.block_L.threshold

    def reset(self):
        self.set_threshold(RESET_THRESHOLD_VALUE)

    def _isolate(self, network=None, threshold=None):
        if network is None:
            network = self.network
        if threshold is None:
            threshold = self._threshold
        return LaplacianSmoothingModel(
            network, self.normalized_belongingness_matrix, threshold,
            neighborhood_walker=self.neighborhood_walker)

    def normalize_belongingness(self, method='colrow'):
        self.normalized_belongingness_matrix = ProportionMatrixNormalization.normalize(
            self.belongingness_matrix, method)
        self._belongingness_normalization = method
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

    def apply_belongingness_patch(self):
        updated_belongingness = self.get_belongingness_patch()
        self.normalized_belongingness_matrix = updated_belongingness
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]

    def remove_belongingness_patch(self):
        self.normalized_belongingness_matrix = ProportionMatrixNormalization.normalize(
            self.belongingness_matrix, self._belongingness_normalization)
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]

    def sample_tau(self, rho, lmda):
        sigma_est = np.std(self.S0)
        mu_tau = self.estimate_tau_from_S0(rho, lmda)
        return np.random.multivariate_normal(mu_tau, np.eye(len(mu_tau)).dot(sigma_est ** 2))

    def sample_phi_given_tau(self, tau, lmda):
        return np.random.multivariate_normal(self.A0.dot(tau), (1. / lmda) * self.L_oo_inv)

    def find_optimal_lambda(self, rho, lambda_max=1, step=0.01, threshold=0.0001, fit_tau=True,
                            drop_missing=True, renormalize_belongingness='colrow'):
        obs = []
        missed = []
        network = self.network.clone()
        for node in network:
            if node.score < threshold:
                missed.append(node)
            else:
                obs.append(node.score)
        lambda_values = np.arange(0.01, lambda_max, step)
        press = []
        if drop_missing:
            for node in missed:
                network.remove_node(node, limit=5)
        wpl = weighted_laplacian_matrix(network)
        lum = LaplacianSmoothingModel(
            network, self.normalized_belongingness_matrix, threshold,
            neighborhood_walker=self.neighborhood_walker,
            belongingness_normalization=renormalize_belongingness)
        ident = np.eye(wpl.shape[0])
        for lambd in lambda_values:
            if fit_tau:
                tau = lum.estimate_tau_from_S0(rho, lambd)
            else:
                tau = np.zeros(self.A0.shape[1])
            T = lum.optimize_observed_scores(lambd, lum.A0.dot(tau))
            A = ident + lambd * wpl
            H = np.linalg.inv(A)
            press_value = sum(
                ((obs - T) / (1 - (np.diag(H) - np.finfo(float).eps))) ** 2) / len(obs)
            press.append(press_value)
        return lambda_values, np.array(press)

    def find_threshold_and_lambda(self, rho, lambda_max=1., lambda_step=0.01, threshold_start=0.,
                                  threshold_step=0.2, fit_tau=True, drop_missing=True,
                                  renormalize_belongingness='colrow'):
        solutions = NetworkReduction()
        limit = max(self.S0)
        start = max(min(self.S0), threshold_start)
        current_network = self.network.clone()
        for threshold in np.arange(start, limit, threshold_step):
            obs = []
            missed = []
            network = current_network.clone()
            for i, node in enumerate(network):
                if node.score < threshold:
                    missed.append(node)
                else:
                    obs.append(node.score)
            if len(obs) == 0:
                break
            obs = np.array(obs)
            lambda_values = np.arange(0.01, lambda_max, lambda_step)
            press = []
            if drop_missing:
                for node in missed:
                    network.remove_node(node, limit=5)
            wpl = weighted_laplacian_matrix(network)
            ident = np.eye(wpl.shape[0])
            lum = LaplacianSmoothingModel(
                network, self.normalized_belongingness_matrix, threshold,
                neighborhood_walker=self.neighborhood_walker,
                belongingness_normalization=renormalize_belongingness)
            lum.apply_belongingness_patch()
            updates = []
            taus = []
            for lambd in lambda_values:
                if fit_tau:
                    tau = lum.estimate_tau_from_S0(rho, lambd)
                else:
                    tau = np.zeros(self.A0.shape[1])
                T = lum.optimize_observed_scores(lambd, lum.A0.dot(tau))
                # Try rescaling
                # T = T / T.max() * obs.max()
                A = ident + lambd * wpl

                H = np.linalg.inv(A)
                diag_H = np.diag(H)
                if len(diag_H) != len(T):
                    diag_H = diag_H[lum.obs_ix]
                    assert len(diag_H) == len(T)

                press_value = sum(
                    ((obs - T) / (1 - (diag_H - np.finfo(float).eps))) ** 2) / len(obs)
                press.append(press_value)
                updates.append(T)
                taus.append(tau)
            solutions[threshold] = NetworkTrimmingSearchSolution(
                threshold, lambda_values, np.array(press), (network), np.array(obs),
                updates, taus, lum)
            current_network = network
        return solutions


class NeighborhoodPrior(object):
    def __init__(self, tau, neighborhood_names):
        self.tau = tau
        self.neighborhood_names = neighborhood_names
        self.prior = OrderedDict(zip(neighborhood_names, tau))

    def __iter__(self):
        return iter(self.tau)

    def items(self):
        return self.prior.items()

    def keys(self):
        return self.prior.keys()

    def __len__(self):
        return len(self.tau)

    def __getitem__(self, i):
        return self.tau[i]

    def getname(self, name):
        return self.prior[name]

    def __repr__(self):
        return "NeighborhoodPrior(%s)" % ([
            "%s: %0.3f" % (n, t) for n, t in self.prior.items()
        ])


class GroupBelongingnessMatrix(object):
    _node_like = (basestring, CompositionGraphNode,
                  GlycanComposition, GlycanCompositionProxy)

    @classmethod
    def from_model(cls, model, normalized=True):
        if normalized:
            mat = model.normalized_belongingness_matrix
        else:
            mat = model.belongingness_matrix
        groups = model.neighborhood_walker.neighborhood_names()
        members = model.network.nodes
        return cls(mat, groups, members)

    def __init__(self, belongingness_matrix, groups, members):
        self.belongingness_matrix = np.array(belongingness_matrix)
        self.groups = [str(x) for x in groups]
        self.members = [str(x) for x in members]
        self._column_indices = OrderedDict([(k, i) for i, k in enumerate(self.groups)])
        self._member_indices = OrderedDict([(k, i) for i, k in enumerate(self.members)])

    def _column_indices_by_name(self, names):
        if isinstance(names, basestring):
            names = [names]
        indices = [self._column_indices[n] for n in names]
        return indices

    def _member_indices_by_name(self, names):
        if isinstance(names, self._node_like):
            names = [names]
        indices = [self._member_indices[n] for n in names]
        return indices

    def _coerce_member(self, names):
        if isinstance(names, self._node_like):
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

    def getindex(self, rows=None, cols=None):
        if rows is not None:
            rows = self._coerce_member(rows)
            row_ix = self._member_indices_by_name(rows)
        else:
            row_ix = None
        if cols is not None:
            cols = self._coerce_column(cols)
            col_ix = self._column_indices_by_name(cols)
        else:
            col_ix = None
        return row_ix, col_ix

    def __getitem__(self, ij):
        return self.belongingness_matrix[ij]

    def __setitem__(self, ij, val):
        self.belongingness_matrix[ij] = val

    def __array__(self):
        return np.array(self.belongingness_matrix)

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


MatrixEditIndex = namedtuple("MatrixEditIndex", ("row_index", "col_index", "action"))
MatrixEditInstruction = namedtuple("MatrixEditInstruction", ("composition", "neighborhood", "action"))


class BelongingnessMatrixPatcher(object):
    def __init__(self, model):
        self.model = model
        self.A0 = model.A0
        self.belongingness_matrix = GroupBelongingnessMatrix.from_model(
            model, normalized=False)

    def find_singleton_neighborhoods(self):
        n_cols = self.A0.shape[1]
        edits = []
        for i in range(n_cols):
            # We have a neighborhood with only one member
            col = self.A0[:, i]
            mask = col > 0
            if mask.sum() == 1:
                j = np.argmax(mask)
                edit = MatrixEditIndex(j, i, 'delete')
                edits.append(edit)
        return edits

    def transform_index_to_key(self, edits):
        neighborhood_names = self.model.neighborhood_walker.neighborhood_names()
        out = []
        for edit in edits:
            key = str(self.model.C0[edit.row_index])
            neighborhood = neighborhood_names[edit.col_index]
            out.append(MatrixEditInstruction(key, neighborhood, edit.action))
        return out

    def patch_belongingness_matrix(self, edits):
        gbm = self.belongingness_matrix
        instructions = self.transform_index_to_key(edits)
        for instruction in instructions:
            ij = gbm.getindex(instruction.composition, instruction.neighborhood)
            if instruction.action == 'delete':
                gbm[ij] = 0
        return gbm

    @classmethod
    def patch(cls, model):
        inst = cls(model)
        targets = inst.find_singleton_neighborhoods()
        out = inst.patch_belongingness_matrix(targets)
        return np.array(out)


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
        ax.set_xlabel("$S_o$ Threshold")
        ax.set_ylabel("Optimal $\lambda$")
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
        self.optimal_lambda = self.lambda_values[np.argmin(self.press_residuals)]
        self.minimum_residuals = self.press_residuals.min()
        self.observed = observed
        self.updated = updated
        self.taus = taus
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


GridPointSolution = namedtuple("GridPointSolution", ("threshold", "lmbda", "tau", "belongingness_matrix"))


class ThresholdSelectionGridSearch(object):
    def __init__(self, model, network_reduction=None, apex_threshold=0.95):
        self.model = model
        self.network_reduction = network_reduction
        self.apex_threshold = apex_threshold

    def explore_grid(self):
        if self.network_reduction is None:
            self.network_reduction = self.model.find_threshold_and_lambda(
                rho=0.1, threshold_step=0.01, fit_tau=True)
        stack = []
        tau_magnitude = []
        xaxis = []

        for level in self.network_reduction:
            if level.threshold < 0.4:
                continue
            stack.append(np.array(level.taus).mean(axis=0))
            tau_magnitude.append(
                (np.abs(level.taus).sum()) + ((level.threshold / 4.) + 0.75)
                # (np.abs(level.taus).sum()) * ((level.threshold / 3.) + (1 - (1. / 3.)))
                )
            xaxis.append(level.threshold)
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

    def _get_estimate_for_state(self, state):
        i = np.argmin(state.press_residuals)
        lmbda = state.lambda_values[i]
        self.model.set_threshold(state.threshold)
        self.model.apply_belongingness_patch()
        tau = self.model.estimate_tau_from_S0(0.1, lmbda)
        A = self.model.normalized_belongingness_matrix.copy()
        self.model.remove_belongingness_patch()
        return GridPointSolution(state.threshold, lmbda, tau, A)

    def get_solutions(self):
        states = self._get_solution_states()
        solutions = [self._get_estimate_for_state(state) for state in states]
        self.model.reset()
        return solutions

    def average_solution(self):
        solutions = self.get_solutions()
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
        A = ProportionMatrixNormalization.normalize(A, self.model._belongingness_normalization)
        return GridPointSolution(thresh_acc, lmbda_acc, tau_acc, A)

    def estimate_phi_observed(self, solution=None, remove_threshold=True):
        if solution is None:
            solution = self.average_solution()
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
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Criterion")
        ax.set_title("Locate Ideal Threshold\nBy Maximizing $\\bar{\\tau_j}$")
        ax.set_xticks([x_ for i, x_ in enumerate(solution.thresholds) if i % 5 == 0])
        ax.set_xticklabels(["%0.2f" % x_ for i, x_ in enumerate(solution.thresholds) if i % 5 == 0])
        return ax

    def plot_thresholds(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        solution = self.explore_grid()
        ax = self.network_reduction.plot(ax)
        ax.vlines(solution.thresholds[solution.apexes], 0, 1, color='red')
        return ax


def smooth_network(network, observed_compositions, threshold_step=0.5, apex_threshold=0.95,
                   belongingness_matrix=None, rho=0.1):
    model = GlycomeModel(observed_compositions, network, belongingness_matrix=belongingness_matrix)
    reduction = model.find_threshold_and_lambda(rho=rho, threshold_step=threshold_step)
    search = ThresholdSelectionGridSearch(model, reduction, apex_threshold)
    phi = search.average_solution()
    return phi, search
