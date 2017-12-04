from collections import OrderedDict, defaultdict

import numpy as np

from glypy import GlycanComposition

from glycan_profiling.database.composition_network import (
    CompositionGraphNode, NeighborhoodWalker)

from .constants import (
    DEFAULT_LAPLACIAN_REGULARIZATION,
    NORMALIZATION,
    DEFAULT_RHO,
    RESET_THRESHOLD_VALUE)

from .laplacian_smoothing import LaplacianSmoothingModel, ProportionMatrixNormalization

from .graph import (
    BlockLaplacian,
    assign_network,
    network_indices,
    weighted_laplacian_matrix)

from .grid_search import (
    NetworkReduction,
    NetworkTrimmingSearchSolution,
    GridPointSolution,
    GridSearchSolution,
    ThresholdSelectionGridSearch)


def _has_glycan_composition(x):
    try:
        gc = x.glycan_composition
        return gc is not None
    except AttributeError:
        return False


class GlycanCompositionSolutionRecord(object):
    def __init__(self, glycan_composition, score, total_signal):
        self.glycan_composition = glycan_composition
        self.score = score
        self.internal_score = self.score
        self.total_signal = total_signal

    @classmethod
    def from_glycan_composition_chromatogram(cls, solution):
        return cls(solution.glycan_composition, solution.score,
                   solution.total_signal)

    def __repr__(self):
        return ("{self.__class__.__name__}({self.glycan_composition}, "
                "{self.score}, {self.total_signal})").format(self=self)


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
                 belongingness_normalization=NORMALIZATION):
        observed_compositions = [
            o for o in observed_compositions if _has_glycan_composition(o)]
        self._observed_compositions = observed_compositions
        self._configure_with_network(network)
        if len(self.miss_ix) == 0:
            self._network.add_node(CompositionGraphNode(GlycanComposition(), -1), reindex=True)
            self._configure_with_network(self._network)

        self.block_L = BlockLaplacian(self.network, regularize=regularize)
        self.threshold = self.block_L.threshold

        # Initialize Names
        self.normalized_belongingness_matrix = None
        self.A0 = None
        self._belongingness_normalization = None
        self.S0 = []
        self.C0 = []
        self.variance_matrix = None

        # Expensive Step
        if belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()
        else:
            self.belongingness_matrix = np.array(belongingness_matrix)

        # Normalize and populate
        self.normalize_belongingness(belongingness_normalization)
        self._populate()

    def _configure_with_network(self, network):
        self._network = network
        self.network = assign_network(network.clone(), self._observed_compositions)

        self.neighborhood_walker = NeighborhoodWalker(self.network)

        self.neighborhood_names = self.neighborhood_walker.neighborhood_names()
        self.node_names = [str(node) for node in self._network]

        self.obs_ix, self.miss_ix = network_indices(self.network)

    def __reduce__(self):
        return self.__class__, (
            self._observed_compositions, self._network, self.belongingness_matrix,
            self.block_L.regularize, self._belongingness_normalization)

    def _populate(self):
        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]
        self.Am = self.normalized_belongingness_matrix[self.miss_ix, :]
        self.S0 = np.array([g.score for g in self.network[self.obs_ix]])
        self.C0 = ([g for g in self.network[self.obs_ix]])
        self.variance_matrix = np.eye(len(self.S0))

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

    def normalize_belongingness(self, method=NORMALIZATION):
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
                            drop_missing=True, renormalize_belongingness=NORMALIZATION):
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
                                  renormalize_belongingness=NORMALIZATION):
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
            updates = []
            taus = []
            for lambd in lambda_values:
                if fit_tau:
                    tau = lum.estimate_tau_from_S0(rho, lambd)
                else:
                    tau = np.zeros(self.A0.shape[1])
                T = lum.optimize_observed_scores(lambd, lum.A0.dot(tau))
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
        return "NeighborhoodPrior(%s)" % ', '.join([
            "%s: %0.3f" % (n, t) for n, t in self.prior.items()
        ])

    def __array__(self):
        return np.array(self.tau)


def smooth_network(network, observed_compositions, threshold_step=0.5, apex_threshold=0.95,
                   belongingness_matrix=None, rho=DEFAULT_RHO, lambda_max=1,
                   include_missing=False, lmbda=None, model_state=None):
    convert = GlycanCompositionSolutionRecord.from_glycan_composition_chromatogram
    observed_compositions = [
        convert(o) for o in observed_compositions if _has_glycan_composition(o)]
    model = GlycomeModel(
        observed_compositions, network,
        belongingness_matrix=belongingness_matrix)
    if model_state is None:
        reduction = model.find_threshold_and_lambda(
            rho=rho, threshold_step=threshold_step,
            lambda_max=lambda_max)
        search = ThresholdSelectionGridSearch(model, reduction, apex_threshold)
        params = search.average_solution(lmbda=lmbda)
    else:
        search = ThresholdSelectionGridSearch(model, None, apex_threshold)
        model_state.reindex(model)
        params = model_state
        if lmbda is not None:
            params.lmbda = lmbda
    network = search.annotate_network(params, include_missing=include_missing)

    return network, search, params


def display_table(names, values, sigfig=3, filter_empty=1, print_fn=None):
    values = np.array(values)
    maxlen = len(max(names, key=len)) + 2
    fstring = ("%%0.%df" % sigfig)
    for i in range(len(values)):
        if values[i, :].sum() or not filter_empty:
            if print_fn is None:
                print(names[i].ljust(maxlen) + ('|'.join([fstring % f for f in values[i, :]])))
            else:
                print_fn(names[i].ljust(maxlen) + ('|'.join([fstring % f for f in values[i, :]])))
