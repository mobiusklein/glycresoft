import numpy as np
import glypy


# Taken from `statsmodels.distribution`

class StepFunction(object):

    def __init__(self, x, y, ival=0., is_sorted=False, side='left'):
        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not is_sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):

    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1. / nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, is_sorted=True)

#


def calculate_relation_matrix(network, phi=2.):
    N = len(network)
    P = np.zeros((N, N))
    for node_i in network.nodes:
        total = np.sum([np.exp(e.order / phi) for e in node_i.edges])
        for edge in node_i.edges:
            j = edge[node_i].index
            i = node_i.index
            P[i, j] = np.exp(edge.order / phi) / total
    return P


def power_iteration(A):
    b = np.zeros(A.shape[0]) + 1. / A.shape[0]
    b0 = b
    while True:
        b = A.dot(b0)
        dist = (np.abs(b - b0).sum() / b0.sum())
        if dist < 1e-4:
            break
        else:
            b0 = b
    return b


class CompositionDistributionUpdater(object):

    def __init__(self, network, solutions, common_weight=1.1, precision_scale=1):
        self.network = network
        self.solutions = solutions
        self.common_weight = common_weight
        self.precision_scale = precision_scale
        self.W = np.array([node.score for node in network.nodes]) * \
            precision_scale + common_weight

        # Dirichlet Mean
        self.pi0 = self.W / self.W.sum()

        self.sol_map = {
            sol.composition: sol.score for sol in solutions if sol.composition is not None}

        self.observations = np.array(
            [self.sol_map.get(node.composition.serialize(), 0) for node in network.nodes])
        self.iterations = 0

    def convergence(self, pi_next, pi_last):
        d = np.abs(pi_last).sum()
        v = (np.abs(pi_next - pi_last).sum()) / d
        return v

    def update_pi(self, pi_array):
        pi2 = np.zeros_like(pi_array)
        total_score_pi = (self.observations * pi_array).sum()
        total_w = ((self.W - 1).sum() + 1)
        for i in range(len(self.W)):
            pi2[i] = ((self.W[i] - 1) + (self.observations[i] *
                                         pi_array[i]) / total_score_pi) / total_w
            assert pi2[i] >= 0, (self.W[i], pi_array[i])
        return pi2

    def optimize_pi(self, pi, maxiter=100, **kwargs):
        pi_last = pi
        pi_next = self.update_pi(pi_last)

        self.iterations = 0
        converging = float('inf')
        while converging > 1e-6 and self.iterations < maxiter:
            pi_last = pi_next
            pi_next = self.update_pi(pi_last)
            self.iterations += 1
            converging = self.convergence(pi_next, pi_last)
        if converging < 1e-6:
            print("Converged in %d iterations" % self.iterations)
        else:
            print("Failed to converge in %d iterations (%f)" % (self.iterations, converging))

        return pi_next

    def fit(self, **kwargs):
        return self.optimize_pi(self.pi0, **kwargs)


class FuzzyCompositionDistributionUpdater(CompositionDistributionUpdater):

    def __init__(self, network, solutions, common_weight=1, precision_scale=1, spread_factor=0.2):
        super(FuzzyCompositionDistributionUpdater, self).__init__(network, solutions, common_weight, precision_scale)
        self.spread_factor = spread_factor

        self.adjacency_list = []
        self.edge_weight_list = []
        for node in self.network.nodes:
            row = []
            weight = []
            for edge in node.edges:
                row.append(edge[node].index)
                weight.append(edge.weight)
            weight = np.array(weight)
            self.adjacency_list.append(row)
            self.edge_weight_list.append(weight)

    def _compute_support(self, pi_array, ix):
        edges = self.adjacency_list[ix]

        weights = self.edge_weight_list[ix]
        support_pis = pi_array[edges]
        support_obs = self.observations[edges]

        norm = (weights * support_obs).sum()
        if norm == 0:
            return 0

        return (support_pis * weights * (support_obs * support_obs)).sum() / norm

    def _compute_score(self, pi_array, ix):
        base = (pi_array[ix]) * (self.observations[ix])
        support_value = self._compute_support(pi_array, ix)
        return base * (1 - self.spread_factor) + support_value * self.spread_factor

    def _compute_fuzzy_pi(self, pi_array):
        out = []
        for i in range(len(pi_array)):
            out.append(self._compute_score(pi_array, i))
        return np.array(out)

    def update_pi(self, pi_array):
        pi2 = np.zeros_like(pi_array)
        fuzzed_pi = self._compute_fuzzy_pi(pi_array)
        total_score_pi = fuzzed_pi.sum()
        total_w = ((self.W - 1).sum() + 1)
        for i in range(len(self.W)):
            pi2[i] = ((self.W[i] - 1) + (fuzzed_pi[i]) / total_score_pi) / total_w
            assert pi2[i] >= 0, (self.W[i], pi_array[i])
        return pi2


def convert_posterior_to_network(posterior, network):
    new = network.clone()
    for i in range(len(new.nodes)):
        node = new.nodes[i]
        score = posterior[i]
        node.score = score
    return new


def null_model_probability_of_shift(cases, mass_shift):
    hits = 0
    total = len(cases)
    for case in cases:
        match = cases.find_mass(case.neutral_mass + mass_shift)
        if match:
            hits += 1
    return hits / float(total)


def valid_model_probability_of_shift(cases, mass_shift, validator):
    hits = 0
    total = 0
    for case in cases:
        if validator(case):
            match = cases.find_mass(case.neutral_mass + mass_shift)
            if match:
                hits += 1
            total += 1
    return hits / float(total)


def probability_of_shift(cases, mass, validator):
    alpha = valid_model_probability_of_shift(cases, mass, validator)
    beta = null_model_probability_of_shift(cases, mass)
    return alpha / (alpha + beta)


def can_gain_fucose(chromatogram):
    comp = chromatogram.composition
    if comp is None:
        return False
    comp = glypy.GlycanComposition.parse(comp)
    nfuc = comp["Fuc"]
    nhexnac = comp['HexNAc']
    return nfuc < nhexnac - 1 and nfuc < 2


def can_lose_fucose(chromatogram):
    comp = chromatogram.composition
    if comp is None:
        return False
    comp = glypy.GlycanComposition.parse(comp)
    nfuc = comp["Fuc"]
    nhexnac = comp['HexNAc']
    return nfuc < nhexnac - 1 and nfuc > 0


def can_gain_neuac(chromatogram):
    comp = chromatogram.composition
    if comp is None:
        return False
    comp = glypy.GlycanComposition.parse(comp)
    nhexnac = comp['HexNAc']
    nneuac = comp["Neu5Ac"]
    return nneuac < nhexnac - 2 and nneuac < 4


def can_lose_neuac(chromatogram):
    comp = chromatogram.composition
    if comp is None:
        return False
    comp = glypy.GlycanComposition.parse(comp)
    nhexnac = comp['HexNAc']
    nneuac = comp["Neu5Ac"]
    return nneuac < nhexnac - 2 and nneuac > 0


def threshold(x, t=0.4):
    return x.score > t
