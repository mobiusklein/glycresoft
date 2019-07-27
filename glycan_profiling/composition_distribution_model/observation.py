from collections import defaultdict, OrderedDict

import numpy as np
from scipy import linalg


class GlycanCompositionSolutionRecord(object):
    def __init__(self, glycan_composition, score, total_signal=1.0):
        self.glycan_composition = glycan_composition
        self.score = score
        self.internal_score = self.score
        self.total_signal = total_signal

    def __eq__(self, other):
        if other is None:
            return False
        match = self.glycan_composition == other.glycan_composition
        if not match:
            return match
        match = np.isclose(self.score, other.score)
        if not match:
            return match
        match = np.isclose(self.total_signal, other.total_signal)
        return match

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.glycan_composition)

    @classmethod
    def from_chromatogram(cls, solution):
        return cls(solution.glycan_composition, solution.score,
                   solution.total_signal)

    def __repr__(self):
        return ("{self.__class__.__name__}({self.glycan_composition}, "
                "{self.score}, {self.total_signal})").format(self=self)


def is_diagonal(m):
    if m.shape[0] != m.shape[1]:
        return False
    return (np.count_nonzero(m) - np.count_nonzero(np.diag(m))) == 0


class ObservationWeightState(object):
    def __init__(self, raw_scores, weight_matrix, observed_indices, size):
        self.raw_scores = np.array(raw_scores)
        self.weight_matrix = weight_matrix
        self.observed_indices = observed_indices
        self.size = size
        self.variance_matrix = None
        self.left_inverse_weight_matrix = None
        self.inverse_variance_matrix = None
        self.weighted_scores = None
        if len(self.raw_scores) == 0:
            self.empty()
        else:
            self.transform()

    def empty(self):
        self.variance_matrix = np.array([[]])
        self.left_inverse_weight_matrix = np.array([[]])
        self.inverse_variance_matrix = np.array([[]])
        self.weighted_scores = np.array([])

    def transform(self):
        # This is necessary when the weight matrix is not a square matrix (e.g. the identity matrix)
        # and it is *very slow*. Consider fast-pathing the identity matrix case.
        w = self.weight_matrix.T.dot(self.weight_matrix)
        if is_diagonal(w):
            winv = np.diag([1 / i if i != 0 else i for i in np.diag(w)])
        else:
            winv = linalg.pinv(w)
        self.left_inverse_weight_matrix = winv.dot(self.weight_matrix.T)
        self.variance_matrix = self.left_inverse_weight_matrix.dot(self.left_inverse_weight_matrix.T)
        self.inverse_variance_matrix = w
        self.weighted_scores = self.left_inverse_weight_matrix.dot(self.raw_scores)
        self.weighted_scores = self.weighted_scores[np.nonzero(self.weighted_scores)]

    def expand_variance_matrix(self):
        V = np.zeros((self.size, self.size))
        V[self.observed_indices, self.observed_indices] = np.diag(self.variance_matrix)
        return V


class VariableObservationAggregation(object):
    def __init__(self, network):
        self.aggregation = defaultdict(list)
        self.network = network

    def collect(self, observations):
        for obs in observations:
            self.aggregation[obs.glycan_composition].append(obs)

    def reset(self):
        self.aggregation = defaultdict(list)

    @property
    def total_observations(self):
        q = 0
        for key, values in self.aggregation.items():
            q += len(values)
        return q

    def iterobservations(self):
        for key, values in sorted(self.aggregation.items(), key=lambda x: self.network[x[0]].index):
            for val in values:
                yield val

    def observed_indices(self):
        indices = {self.network[obs.glycan_composition].index for obs in self.iterobservations()}
        return np.array(sorted(indices))

    def calculate_weight(self, observation):
        return 1

    def build_weight_matrix(self):
        q = self.total_observations
        p = len(self.network)
        weights = np.zeros((q, p))
        for i, obs in enumerate(self.iterobservations()):
            weights[i, self.network[obs.glycan_composition].index] = self.calculate_weight(obs)
        return weights

    def estimate_summaries(self):
        E = self.build_weight_matrix()
        scores = [r.score for r in self.iterobservations()]
        return ObservationWeightState(scores, E, self.observed_indices(), len(self.network))

    def build_records(self):
        observation_weights = self.estimate_summaries()
        indices = self.observed_indices()
        nodes = self.network[indices]
        records = []
        indices = []
        for i, node in enumerate(nodes):
            rec = GlycanCompositionSolutionRecord(
                node.glycan_composition, observation_weights.weighted_scores[i],
                sum([rec.total_signal for rec in self.aggregation[node.glycan_composition]]),
            )
            records.append(rec)
            indices.append(node.index)
        return records, observation_weights


class AbundanceWeightedObservationAggregation(VariableObservationAggregation):
    def calculate_weight(self, observation):
        return np.log10(observation.total_signal) / np.log10(1e6)
