import numpy as np

from glycan_profiling.task import TaskBase


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
