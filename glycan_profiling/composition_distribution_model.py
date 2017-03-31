import numpy as np
from scipy import linalg

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


def assign_network(network, observed):
    rns = network_scoring.RenormalizingNetworkScorerDifferenceMethod(
        observed, network.clone())
    rns.build_solution_map()
    rns.assign_network()
    return rns.network


def network_indices(network, threshold=0.0001):
    missing = []
    observed = []
    for node in network:
        if node._temp_score < threshold:
            missing.append(node.index)
        else:
            observed.append(node.index)
    return observed, missing


def make_blocks(network, observed, lmbda=0.2, threshold=0.4):
    network = assign_network(network, observed)
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
    update_obs_scores = optimize_observed_scores(blocks, lmbda, observed_scores, t0)
    update_missing_scores = compute_missing_scores(blocks, update_obs_scores, t0, tm)

    if fit_t:
        for node, score in zip(observed_labels, update_obs_scores):
            node.score = score
        nhw = composition_network.NeighborhoodWalker(network)
        nhann = composition_network.DistanceWeightedNeighborhoodAnalyzer(nhw)
        t0_update = []
        for node in observed_labels:
            t0_update.append(nhann[node])
        update_obs_scores = optimize_observed_scores(blocks, lmbda, observed_scores, t0_update)
        update_missing_scores = compute_missing_scores(blocks, update_obs_scores, t0_update, tm)

    return update_obs_scores, update_missing_scores, observed_labels, observed_scores
