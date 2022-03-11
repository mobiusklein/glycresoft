import numpy as np
from scipy import stats
from scipy.special import logsumexp

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


class KMeans(object):

    def __init__(self, k, means=None):
        self.k = k
        self.means = np.array(means) if means is not None else None

    @classmethod
    def fit(cls, X, k):
        mus = np.sort(np.random.choice(X, k))
        inst = cls(k, mus)
        inst.estimate(X)
        return inst

    def estimate(self, X, maxiter=1000, tol=1e-6):
        for i in range(maxiter):
            distances = []
            for k in range(self.k):
                diff = (X - self.means[k])
                dist = np.sqrt((diff * diff))
                distances.append(dist)
            distances = np.vstack(distances).T
            cluster_assignments = np.argmin(distances, axis=1)
            new_means = []
            for k in range(self.k):
                new_means.append(
                    np.mean(X[cluster_assignments == k]))
            new_means = np.array(new_means)
            new_means[np.isnan(new_means)] = 0.0
            diff = (self.means - new_means)
            dist = np.sqrt((diff * diff).sum()) / self.k
            self.means = new_means
            if dist < tol:
                break
        else:
            pass

    def score(self, x):
        x = np.asanyarray(x)
        if x.ndim < 2:
            x = x.reshape((-1, 1))
        delta = np.abs(self.means - x)
        score = 1 - delta
        score /= score.sum(axis=1)[:, None]
        return score

    def predict(self, x):
        scores = self.score(x)
        return np.argmax(scores, axis=1)


class MixtureBase(object):
    def __init__(self, n_components):
        self.n_components

    def loglikelihood(self, X):
        out = logsumexp(self.logpdf(X), axis=1).sum()
        return out

    def bic(self, X):
        '''Calculate the Bayesian Information Criterion
        for selecting the most parsimonious number of components.
        '''
        return np.log(X.size) * (self.n_components * 3 - 1) - (2 * (self.loglikelihood(X)))

    def logpdf(self, X, weighted=True):
        out = np.array(
            [self._logpdf(X, k)
             for k in range(self.n_components)]).T
        if weighted:
            out += np.log(self.weights)
        return out

    def pdf(self, X, weighted=True):
        return np.exp(self.logpdf(X, weighted=weighted))

    def score(self, X):
        return self.pdf(X).sum(axis=1)

    def responsibility(self, X):
        '''Also called the posterior probability, as these are the
        probabilities associating each element of X with each component
        '''
        acc = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            acc[:, k] = np.log(self.weights[k]) + self._logpdf(X, k)
        total = logsumexp(acc, axis=1)[:, None]
        # compute the ratio of the density to the total in log-space, then
        # exponentiate to return to linear space
        out = np.exp(acc - total)
        return out


class GaussianMixture(MixtureBase):
    def __init__(self, mus, sigmas, weights):
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.weights = np.array(weights)
        self.n_components = len(weights)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.mus}, {self.sigmas}, {self.weights})"
        return template.format(self=self)

    def _logpdf(self, X, k):
        '''Computes the log-space density for `X` using the `k`th
        component of the mixture
        '''
        return stats.norm.logpdf(X, self.mus[k], self.sigmas[k])

    @classmethod
    def fit(cls, X, n_components, maxiter=1000, tol=1e-5, deterministic=True):
        if not deterministic:
            mus = KMeans.fit(X, n_components).means
        else:
            mus = (np.max(X) / (n_components + 1)) * np.arange(1, n_components + 1)
        assert not np.any(np.isnan(mus))
        sigmas = np.var(X) * np.ones_like(mus)
        weights = np.ones_like(mus) / n_components
        inst = cls(mus, sigmas, weights)
        inst.estimate(X, maxiter=maxiter, tol=tol)
        return inst

    def _update_params_for(self, X, k, responsibility, new_mus, new_sigmas, new_weights):
        # The expressions for each partial derivative may be useful for understanding
        # portions of this block.
        # See http://www.notenoughthoughts.net/posts/normal-log-likelihood-gradient.html
        g = responsibility[:, k]
        N_k = g.sum()
        # Begin specialization for Gaussian distributions
        diff = X - self.mus[k]
        mu_k = g.dot(X) / N_k
        new_mus[k] = mu_k
        sigma_k = (g * diff).dot(diff.T) / N_k + 1e-6
        new_sigmas[k] = np.sqrt(sigma_k)
        new_weights[k] = N_k

    def estimate(self, X, maxiter=1000, tol=1e-5):
        for i in range(maxiter):
            # E-step
            responsibility = self.responsibility(X)

            # M-step
            new_mus = np.zeros_like(self.mus)
            new_sigmas = np.zeros_like(self.sigmas)
            prev_loglikelihood = self.loglikelihood(X)
            new_weights = np.zeros_like(self.weights)

            for k in range(self.n_components):
                self._update_params_for(X, k, responsibility, new_mus, new_sigmas, new_weights)

            new_weights /= new_weights.sum()
            self.mus = new_mus
            self.sigmas = new_sigmas
            self.weights = new_weights
            new_loglikelihood = self.loglikelihood(X)
            delta_fit = (prev_loglikelihood - new_loglikelihood) / new_loglikelihood
            if abs(delta_fit) < tol:
                break
        else:
            pass

    @property
    def domain(self):
        return [
            self.mus.min() - self.sigmas.max() * 4,
            self.mus.max() + self.sigmas.max() * 4
         ]

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = np.arange(*self.domain, step=0.01)
        Y = np.exp(self.logpdf(X, True))
        ax.plot(X, np.sum(Y, axis=1), **kwargs)
        return ax


a = 0.0
b = float('inf')
phi_a = stats.norm.pdf(a)
phi_b = stats.norm.pdf(b)
Z = stats.norm.cdf(b) - stats.norm.cdf(a)


def _truncnorm_mean(X):
    mu = X.mean()
    sigma = np.std(X)
    return mu + (phi_a / Z) * sigma


def _truncnorm_std(X):
    sigma2 = np.var(X)
    return np.sqrt(sigma2 * (1 + a * phi_a / Z - (phi_a / Z) ** 2))


def truncnorm_pdf(x, mu, sigma):
    scalar = np.isscalar(x)
    if scalar:
        x = np.array([x])
    mask = (a <= x) & (x <= b)
    out = np.zeros_like(x)

    numerator = stats.norm.pdf((x[mask] - mu) / sigma)
    denominator = stats.norm.cdf(
        (b - mu) / sigma) - stats.norm.cdf((a - mu) / sigma)
    out[mask] = 1 / sigma * numerator / denominator
    if scalar:
        out = out[0]
    return out


def truncnorm_logpdf(x, mu, sigma):
    return np.log(truncnorm_pdf(x, mu, sigma))


class _TruncatedNormalMixin(object):

    def _logpdf(self, X, k):
        '''Computes the log-space density for `X` using the `k`th
        component of the mixture
        '''
        result = truncnorm_logpdf(
            X, self.mus[k], self.sigmas[k])
        return result

    def _update_params_for(self, X, k, responsibility, new_mus, new_sigmas, new_weights):
        # The expressions for each partial derivative may be useful for understanding
        # portions of this block.
        # See http://www.notenoughthoughts.net/posts/normal-log-likelihood-gradient.html
        g = responsibility[:, k]
        N_k = g.sum()

        # Begin specialization for the truncated Gaussian distributions
        diff = X - self.mus[k]

        unconstrained_mu_k = g.dot(X) / N_k
        unconstrained_sigma_k = np.sqrt((g * diff).dot(diff.T) / N_k + 1e-6)
        mu_k = unconstrained_mu_k + (phi_a / Z) * unconstrained_sigma_k
        sigma_k = np.sqrt(unconstrained_sigma_k ** 2 * (1 + a * phi_a / Z - (phi_a / Z) ** 2))

        new_mus[k] = mu_k
        new_sigmas[k] = np.sqrt(sigma_k)
        new_weights[k] = N_k


class TruncatedGaussianMixture(_TruncatedNormalMixin, GaussianMixture):
    @classmethod
    def fit(cls, X, n_components, maxiter=1000, tol=1e-5, deterministic=True):
        if not deterministic:
            mus = KMeans.fit(X, n_components).means
        else:
            mus = (np.max(X) / (n_components + 1)) * \
                np.arange(0, n_components + 1)
        mus[0] = 0.0
        assert not np.any(np.isnan(mus))
        sigmas = np.var(X) * np.ones_like(mus)
        weights = np.ones_like(mus) / n_components
        inst = cls(mus, sigmas, weights)
        inst.estimate(X, maxiter=maxiter, tol=tol)
        return inst


class GammaMixtureBase(MixtureBase):
    def __init__(self, shapes, scales, weights):
        self.shapes = np.array(shapes)
        self.scales = np.array(scales)
        self.weights = np.array(weights)
        self.n_components = len(weights)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.shapes}, {self.scales}, {self.weights})"
        return template.format(self=self)

    def _logpdf(self, X, k):
        '''Computes the log-space density for `X` using the `k`th
        component of the mixture
        '''
        return stats.gamma.logpdf(X, a=self.shapes[k], scale=self.scales[k])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = np.arange(*self.domain, step=0.01)
        Y = np.exp(self.logpdf(X, True))
        ax.plot(X, np.sum(Y, axis=1), **kwargs)
        return ax

    @property
    def domain(self):
        return [
            1e-6,
            100.0
        ]

    @classmethod
    def fit(cls, X, n_components, maxiter=100, tol=1e-5, deterministic=True):
        shapes, scales, weights = cls.initial_parameters(X, n_components, deterministic=deterministic)
        inst = cls(shapes, scales, weights)
        inst.estimate(X, maxiter=maxiter, tol=tol)
        return inst


class IterativeGammaMixture(GammaMixtureBase):
    '''An iterative approximation of a mixture Gamma distributions
    based on the Gaussian distribution. May not converge to the optimal
    solution, and if so, it converges slowly.

    Derived from pGlyco's FDR estimation method
    '''
    @staticmethod
    def initial_parameters(X, n_components, deterministic=True):
        mu = np.median(X) / (n_components + 1) * np.arange(1, n_components + 1)
        sigma = np.ones(n_components) * np.var(X)
        shapes = mu ** 2 / sigma
        scales = sigma / mu
        weights = np.ones(n_components)
        weights /= weights.sum()
        return shapes, scales, weights

    def estimate(self, X, maxiter=100, tol=1e-5):
        prev_loglikelihood = self.loglikelihood(X)
        for i in range(maxiter):
            # E-Step
            responsibility = self.responsibility(X)

            # M-Step
            new_weights = responsibility.sum(axis=0) / responsibility.sum()
            mu = responsibility.T.dot(X) / responsibility.T.sum(axis=1) + 1e-6
            sigma = np.array(
                [responsibility[:, i].dot((X - mu[i]) ** 2 / np.sum(responsibility[:, i]))
                 for i in range(self.n_components)]) + 1e-6
            new_shapes = mu ** 2 / sigma
            new_scales = sigma / mu
            self.shapes = new_shapes
            self.scales = new_scales
            self.weights = new_weights

            new_loglikelihood = self.loglikelihood(X)
            delta_fit = (prev_loglikelihood - new_loglikelihood) / new_loglikelihood
            if abs(delta_fit) < tol:
                break
        else:
            pass


GammaMixture = IterativeGammaMixture


class GaussianMixtureWithPriorComponent(GaussianMixture):
    def __init__(self, mus, sigmas, prior, weights):
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.prior = prior
        self.weights = np.array(weights)
        self.n_components = len(weights)

    def _logpdf(self, X, k):
        if k == self.n_components - 1:
            return np.log(np.exp(self.prior.logpdf(X, weighted=False)).dot(self.prior.weights))
        else:
            return super(GaussianMixtureWithPriorComponent, self)._logpdf(X, k)

    @classmethod
    def fit(cls, X, n_components, prior, maxiter=1000, tol=1e-5, deterministic=True):
        if not deterministic:
            mus = KMeans.fit(X, n_components).means
        else:
            mus = (np.max(X) / (n_components + 1)) * np.arange(1, n_components + 1)
        assert not np.any(np.isnan(mus))
        sigmas = np.var(X) * np.ones_like(mus)
        weights = np.ones(n_components + 1) / (n_components + 1)
        inst = cls(mus, sigmas, prior, weights)
        inst.estimate(X, maxiter=maxiter, tol=tol)
        return inst

    def estimate(self, X, maxiter=1000, tol=1e-5):
        for i in range(maxiter):
            # E-step
            responsibility = self.responsibility(X)

            # M-step
            new_mus = np.zeros_like(self.mus)
            new_sigmas = np.zeros_like(self.sigmas)
            prev_loglikelihood = self.loglikelihood(X)
            new_weights = np.zeros_like(self.weights)
            for k in range(self.n_components - 1):
                self._update_params_for(
                    X, k, responsibility, new_mus, new_sigmas, new_weights)

            new_weights = responsibility.sum(axis=0) / responsibility.sum()
            self.mus = new_mus
            self.sigmas = new_sigmas
            self.weights = new_weights
            new_loglikelihood = self.loglikelihood(X)
            delta_fit = (prev_loglikelihood - new_loglikelihood) / new_loglikelihood
            if abs(delta_fit) < tol:
                break
        else:
            pass

    def plot(self, ax=None, **kwargs):
        ax = super(GaussianMixtureWithPriorComponent, self).plot(ax=ax, **kwargs)
        X = np.arange(self.prior.domain[0], self.mus.max() + self.sigmas.max() * 4, 0.01)
        Y = self.prior.score(X) * self.weights[-1]
        ax.plot(X, Y, **kwargs)
        return ax


class TruncatedGaussianMixtureWithPriorComponent(_TruncatedNormalMixin, GaussianMixtureWithPriorComponent):

    @classmethod
    def fit(cls, X, n_components, prior, maxiter=1000, tol=1e-5, deterministic=True):
        if not deterministic:
            mus = KMeans.fit(X, n_components).means
        else:
            mus = (np.max(X) / (n_components + 1)) * \
                np.arange(1, n_components + 1)
        assert not np.any(np.isnan(mus))
        mus[0] = 0.0
        sigmas = np.var(X) * np.ones_like(mus)
        weights = np.ones(n_components + 1) / (n_components + 1)
        inst = cls(mus, sigmas, prior, weights)
        inst.estimate(X, maxiter=maxiter, tol=tol)
        return inst

    def _logpdf(self, X, k):
        if k == self.n_components - 1:
            return np.log(np.exp(self.prior.logpdf(X, weighted=False)).dot(self.prior.weights))
        else:
            return _TruncatedNormalMixin._logpdf(self, X, k)

    def _update_params_for(self, X, k, responsibility, new_mus, new_sigmas, new_weights):
        # The expressions for each partial derivative may be useful for understanding
        # portions of this block.
        # See http://www.notenoughthoughts.net/posts/normal-log-likelihood-gradient.html
        g = responsibility[:, k]
        N_k = g.sum()
        # Begin specialization for Gaussian distributions
        diff = X - self.mus[k]
        if k == 0:
            mu_k = 0.0
        else:
            mu_k = g.dot(X) / N_k
        new_mus[k] = mu_k
        sigma_k = (g * diff).dot(diff.T) / N_k + 1e-6
        new_sigmas[k] = np.sqrt(sigma_k)
        new_weights[k] = N_k
