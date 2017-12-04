import numpy as np
from scipy import linalg

from .constants import DEFAULT_LAPLACIAN_REGULARIZATION


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
