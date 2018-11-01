from collections import OrderedDict, defaultdict

import numpy as np
from scipy import linalg

from glypy import GlycanComposition

from glycan_profiling.database.composition_network import (
    CompositionGraphNode, NeighborhoodWalker)

from glycan_profiling.task import log_handle

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
    ThresholdSelectionGridSearch)

from .observation import (
    GlycanCompositionSolutionRecord,
    VariableObservationAggregation,
    ObservationWeightState)


def _has_glycan_composition(x):
    try:
        gc = x.glycan_composition
        return gc is not None
    except AttributeError:
        return False


class GlycomeModel(LaplacianSmoothingModel):

    def __init__(self, observed_compositions, network, belongingness_matrix=None,
                 regularize=DEFAULT_LAPLACIAN_REGULARIZATION,
                 belongingness_normalization=NORMALIZATION,
                 observation_aggregator=VariableObservationAggregation):
        self.observation_aggregator = observation_aggregator
        observed_compositions = [
            o for o in observed_compositions if _has_glycan_composition(o) and o.score > 0]
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
        self.S0 = np.array([])
        self.C0 = []
        self.variance_matrix = None
        self.inverse_variance_matrix = None

        # Expensive Step
        if belongingness_matrix is None:
            self.belongingness_matrix = self.build_belongingness_matrix()
        else:
            self.belongingness_matrix = np.array(belongingness_matrix)

        # Normalize and populate
        self.normalize_belongingness(belongingness_normalization)
        self._populate(self._observed_compositions)

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
            self.block_L.regularize, self._belongingness_normalization,
            self.observation_aggregator)

    def _populate(self, observations):
        var_agg = self.observation_aggregator(self._network)
        var_agg.collect(observations)
        aggregated_observations, summarized_state = var_agg.build_records()
        self.network = assign_network(self._network.clone(), aggregated_observations)
        self.obs_ix, self.miss_ix = network_indices(self.network)

        self.A0 = self.normalized_belongingness_matrix[self.obs_ix, :]
        self.Am = self.normalized_belongingness_matrix[self.miss_ix, :]
        self.S0 = np.array([g.score for g in self.network[self.obs_ix]])
        self.C0 = ([g for g in self.network[self.obs_ix]])
        self.summarized_state = summarized_state
        self.variance_matrix = np.diag(summarized_state.variance_matrix[self.obs_ix, self.obs_ix])
        self.inverse_variance_matrix = np.diag(summarized_state.inverse_variance_matrix[self.obs_ix, self.obs_ix])

    def set_threshold(self, threshold):
        accepted = [
            g for g in self._observed_compositions if g.score > threshold]
        if len(accepted) == 0:
            raise ValueError("Threshold %f produces an empty observed set" % (threshold,))
        self._populate(accepted)
        self.block_L = BlockLaplacian(self.network, threshold=threshold, regularize=self.block_L.regularize)
        self.threshold = self.block_L.threshold

    def reset(self):
        self.set_threshold(RESET_THRESHOLD_VALUE)

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
                node.marked = True
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
            belongingness_normalization=renormalize_belongingness,
            variance_matrix=self.variance_matrix)
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

    def find_threshold_and_lambda(self, rho, lambda_max=1., lambda_step=0.02, threshold_start=0.,
                                  threshold_step=0.2, fit_tau=True, drop_missing=True,
                                  renormalize_belongingness=NORMALIZATION):
        r'''Iterate over score thresholds and smoothing factors (lambda), sampling points
        from the parameter grid and computing the PRESS residual at each point.

        This produces a :class:`NetworkReduction` data structure recording the results for
        later local maximum detection.

        Parameters
        ----------
        rho: float
            The scale of the variance of the observed score
        lambda_max: float
            The maximum value of lambda to consider on the grid
        lambda_step: float
            The size of the change in lambda at each iteration
        threshold_start: float
            The minimum observed score threshold to start the grid search at
        threshold_step: float
            The size of the change in the observed score threshold at each iteration
        fit_tau: bool
            Whether or not to estimate :math:`\tau` for each iteration when computing
            the PRESS
        drop_missing: bool
            Whether or not to remove nodes from the graph which are not observed above
            the threshold, restructuring the graph, which in turn changes the Laplacian.
        renormalize_belongingness: str
            A string constant which names the belongingness normalization technique to
            use.

        Returns
        -------
        :class:`NetworkReduction`:
            The recorded grid of sampled points and snapshots of the model at each point
        '''
        solutions = NetworkReduction()
        limit = max(self.S0)
        start = max(min(self.S0) - 1e-3, threshold_start)
        current_network = self.network.clone()
        thresholds = np.arange(start, limit, threshold_step)
        last_solution = None
        for i_threshold, threshold in enumerate(thresholds):
            if i_threshold % 10 == 0:
                log_handle.log("... Threshold = %r (%0.2f%%)" % (
                    threshold, (100.0 * i_threshold / len(thresholds))))
            # Aggregate the raw observations into averaged, variance reduced records
            # and annotate the network with these new scores
            raw_observations = [c for c in self._observed_compositions if c.score > threshold]
            agg = self.observation_aggregator(self.network)
            agg.collect(raw_observations)

            observations, summarized_state = agg.build_records()
            variance_matrix = summarized_state.variance_matrix
            inverse_variance_matrix = summarized_state.inverse_variance_matrix
            obs_ix = agg.observed_indices()
            variance_matrix = np.diag(variance_matrix[obs_ix, obs_ix])
            inverse_variance_matrix = np.diag(inverse_variance_matrix[obs_ix, obs_ix])

            # clear the scores from the network
            current_network = current_network.clone()
            for i, node in enumerate(current_network):
                node.score = 0
            # assign aggregated scores to the network
            network = assign_network(current_network, observations)

            # Filter the network, marking nodes for removal and recording observed
            # nodes for future use.
            obs = []
            missed = []
            for i, node in enumerate(network):
                if node.score < threshold:
                    missed.append(node)
                    node.marked = True
                else:
                    obs.append(node.score)
            if len(obs) == 0:
                break
            obs = np.array(obs)
            press = []

            if drop_missing:
                # drop nodes whose score does not exceed the threshold
                for node in missed:
                    network.remove_node(node, limit=5)

            if last_solution is not None:
                # If after pruning the network, no new nodes have been removed,
                # the optimal solution won't have changed from previous iteration
                # so just reuse the solution
                if last_solution.network == network:
                    current_solution = last_solution.copy()
                    current_solution.threshold = threshold
                    solutions[threshold] = current_solution
                    last_solution = current_solution
                    current_network = network
                    continue
            wpl = weighted_laplacian_matrix(network)
            ident = np.eye(wpl.shape[0])
            lum = LaplacianSmoothingModel(
                network, self.normalized_belongingness_matrix, threshold,
                neighborhood_walker=self.neighborhood_walker,
                belongingness_normalization=renormalize_belongingness,
                variance_matrix=variance_matrix,
                inverse_variance_matrix=inverse_variance_matrix)
            updates = []
            taus = []
            lambda_values = np.arange(0.01, lambda_max, lambda_step)
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
            current_solution = NetworkTrimmingSearchSolution(
                threshold, lambda_values, np.array(press), network, np.array(obs),
                updates, taus, lum)

            solutions[threshold] = current_solution
            last_solution = current_solution
            current_network = network
        return solutions


def smooth_network(network, observed_compositions, threshold_step=0.5, apex_threshold=0.95,
                   belongingness_matrix=None, rho=DEFAULT_RHO, lambda_max=1,
                   include_missing=False, lmbda=None, model_state=None,
                   observation_aggregator=VariableObservationAggregation,
                   belongingness_normalization=NORMALIZATION):
    convert = GlycanCompositionSolutionRecord.from_chromatogram
    observed_compositions = [
        convert(o) for o in observed_compositions if _has_glycan_composition(o)]
    model = GlycomeModel(
        observed_compositions, network,
        belongingness_matrix=belongingness_matrix,
        observation_aggregator=observation_aggregator,
        belongingness_normalization=belongingness_normalization)
    log_handle.log("... Begin Model Fitting")
    if model_state is None:
        reduction = model.find_threshold_and_lambda(
            rho=rho, threshold_step=threshold_step,
            lambda_max=lambda_max)
        if len(reduction) == 0:
            log_handle.log("... No Network Reduction Found")
            return None, None, None
        search = ThresholdSelectionGridSearch(model, reduction, apex_threshold)
        params = search.average_solution(lmbda=lmbda)
    else:
        search = ThresholdSelectionGridSearch(model, None, apex_threshold)
        model_state.reindex(model)
        params = model_state
        if lmbda is not None:
            params.lmbda = lmbda
    log_handle.log("... Projecting Solution Onto Network")
    network = search.annotate_network(params, include_missing=include_missing)

    return network, search, params
