import numpy as np


def gauss(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


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

    def __repr__(self):
        return "MassAccuracyModel(%e, %e)" % (self.mu / self.scale, self.sigma / self.scale)


class MassAccuracyScorer(object):
    def __init__(self, target_model, decoy_model):
        self.target_model = target_model
        self.decoy_model = decoy_model

    def score(self, error):
        target = self.target_model.score(error)
        decoy = self.decoy_model.score(error)
        return target / (target + decoy)

    def __repr__(self):
        return "MassAccuracyScorer(%r, %r)" % (self.target_model, self.decoy_model)

    @classmethod
    def fit(cls, target_matches, decoy_matches):
        return cls(
            MassAccuracyModel.fit(target_matches),
            MassAccuracyModel.fit(decoy_matches))
