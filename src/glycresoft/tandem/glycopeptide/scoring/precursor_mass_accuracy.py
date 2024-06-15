import math
import numpy as np


sqrt2pi = np.sqrt(2 * np.pi)


def gauss(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sqrt2pi)


class MassAccuracyModel(object):

    @classmethod
    def fit(cls, matches):
        obs = [x.best_solution().precursor_mass_accuracy() for x in matches if len(x) > 0]
        mu = np.mean(obs)
        sigma = np.std(obs)
        return cls(mu, sigma)

    def __init__(self, mu, sigma, scale=1e6):
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.mu *= self.scale
        self.sigma *= self.scale

    def _sample(self):
        return self.score(self._interval(n_std=3))

    def _interval(self, n_std=3):
        w = self.sigma * n_std
        return np.arange(self.mu - w, self.mu + w, 1) / self.scale

    def score(self, error):
        return gauss(self.scale * error, self.mu, self.sigma)

    def __call__(self, error):
        return self.score(error)

    def __repr__(self):
        return "MassAccuracyModel(%e, %e)" % (self.mu / self.scale, self.sigma / self.scale)


class MassAccuracyMixin(object):
    accuracy_bias = MassAccuracyModel(0, 5e-6)

    def _precursor_mass_accuracy_score(self):
        offset, error = self.determine_precursor_offset(include_error=True)
        mass_accuracy = -10 * math.log10(1 - self.accuracy_bias(error))
        return mass_accuracy


try:
    from glycresoft._c.tandem.tandem_scoring_helpers import gauss
except ImportError:
    pass
