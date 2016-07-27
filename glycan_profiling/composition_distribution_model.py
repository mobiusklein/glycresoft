import numpy as np


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


def optimize_composition_distribution(network, solutions, x=1, y=1):
    W = np.array([node.score if node.score != 0 else 0 for node in network.nodes]) * y + x

    min_shift = 0
    W[W < W.min() + min_shift] = W.min() + min_shift

    # Dirichlet Mean
    pi = W / W.sum()

    sol_map = {sol.composition: sol.score for sol in solutions if sol.composition is not None}

    S = np.array([sol_map.get(node.composition.serialize(), 1e-15) for node in network.nodes])

    def update_pi(pi_array):
        pi2 = np.zeros_like(pi_array)
        total_score_pi = (S * pi_array).sum()
        total_w = ((W - 1).sum() + 1)
        for i in range(len(W)):
            pi2[i] = ((W[i] - 1) + (S[i] * pi_array[i]) / total_score_pi) / total_w
            assert pi2[i] > 0, (W[i], pi_array[i])
        return pi2

    def optimize_pi(pi, maxiter=100):
        pi_last = pi
        pi_next = update_pi(pi_last)

        def convergence():
            d = np.abs(pi_last).sum()
            v = (np.abs(pi_next).sum() - d) / d
            return v

        i = 0
        while convergence() > 1e-16 and i < maxiter:
            pi_last = pi_next
            pi_next = update_pi(pi_last)
            i += 1

        return pi_next

    return optimize_pi(pi)
