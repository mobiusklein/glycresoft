from array import ArrayType as array
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, Optional, Protocol, Union, List

import numpy as np
from numpy.typing import ArrayLike

from scipy import stats
from scipy.special import logsumexp

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


class KMeans(object):
    k: int
    means: np.ndarray

    def __init__(self, k, means=None):
        self.k = k
        self.means = np.array(means) if means is not None else None

    @classmethod
    def fit(cls, X: np.ndarray, k: int, initial_mus: Optional[np.ndarray]=None):
        if initial_mus is not None:
            mus = initial_mus
        else:
            mus = np.sort(np.random.choice(X, k))
        inst = cls(k, mus)
        inst.estimate(X)
        return inst

    @classmethod
    def from_json(cls, state: Dict) -> 'KMeans':
        return cls(state['k'], state['means'])

    def to_json(self) -> Dict:
        return {
            "k": int(self.k),
            "means": [float(m) for m in self.means]
        }

    def estimate(self, X: np.ndarray, maxiter: int=1000, tol: float=1e-6):
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
            # Always sort the means so that ordering is consistent
            self.means.sort()
            if dist < tol:
                break
        else:
            pass

    def score(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x = np.asanyarray(x)
        if x.ndim < 2:
            x = x.reshape((-1, 1))
        delta = np.abs(self.means - x)
        score = 1 - delta
        norm = score.sum(axis=1)[:, None] + 1e-13
        score /= norm
        return score

    def predict(self, x: Union[float, np.ndarray]) -> np.ndarray:
        scores = self.score(x)
        return np.argmax(scores, axis=1)


class MixtureBase(object):
    n_components: int

    def __init__(self, n_components):
        self.n_components

    def to_json(self) -> Dict:
        return {}

    @classmethod
    def from_json(cls, state) -> 'MixtureBase':
        raise NotImplementedError()

    def loglikelihood(self, X) -> float:
        out = logsumexp(self.logpdf(X), axis=1).sum()
        return out

    def bic(self, X) -> float:
        '''Calculate the Bayesian Information Criterion
        for selecting the most parsimonious number of components.
        '''
        return np.log(X.size) * (self.n_components * 3 - 1) - (2 * (self.loglikelihood(X)))

    def logpdf(self, X: np.ndarray, weighted: float=True) -> np.ndarray:
        out = np.array(
            [self._logpdf(X, k)
             for k in range(self.n_components)]).T
        if weighted:
            out += np.log(self.weights)
        return out

    def pdf(self, X: np.ndarray, weighted: float = True) -> np.ndarray:
        return np.exp(self.logpdf(X, weighted=weighted))

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.pdf(X).sum(axis=1)

    def responsibility(self, X: np.ndarray) -> np.ndarray:
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
    mus: np.ndarray
    sigmas: np.ndarray
    weights: np.ndarray

    def __init__(self, mus, sigmas, weights):
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.weights = np.array(weights)
        self.n_components = len(weights)

    @classmethod
    def from_json(cls, state: Dict) -> 'GaussianMixture':
        return cls(state['mus'], state['sigmas'], state['weights'])

    def to_json(self) -> Dict:
        return {
            "mus": [float(m) for m in self.mus],
            "sigmas": [float(m) for m in self.sigmas],
            "weights": [float(m) for m in self.weights]
        }

    def __repr__(self):
        template = "{self.__class__.__name__}({self.mus}, {self.sigmas}, {self.weights})"
        return template.format(self=self)

    def _logpdf(self, X: np.ndarray, k: int) -> np.ndarray:
        '''Computes the log-space density for `X` using the `k`th
        component of the mixture
        '''
        return stats.norm.logpdf(X, self.mus[k], self.sigmas[k])

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int, maxiter: int=1000, tol: float=1e-5, deterministic: bool=True) -> 'GaussianMixture':
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

    def _update_params_for(self, X: np.ndarray, k: int, responsibility: np.ndarray, new_mus: np.ndarray,
                           new_sigmas: np.ndarray, new_weights: np.ndarray):
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

    def estimate(self, X: np.ndarray, maxiter: int=1000, tol: float=1e-5):
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

    def _logpdf(self, X: np.ndarray, k: int) -> np.ndarray:
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
    def fit(cls, X, n_components, maxiter=1000, tol=1e-5, deterministic=True) -> 'TruncatedGaussianMixture':
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
    shapes: np.ndarray
    scales: np.ndarray
    weights: np.ndarray

    def __init__(self, shapes, scales, weights):
        self.shapes = np.array(shapes)
        self.scales = np.array(scales)
        self.weights = np.array(weights)
        self.n_components = len(weights)

    @classmethod
    def from_json(cls, state: Dict) -> 'GammaMixtureBase':
        return cls(state['shapes'], state['scales'], state['weights'])

    def to_json(self) -> Dict:
        return {
            "shapes": [float(m) for m in self.shapes],
            "scales": [float(m) for m in self.scales],
            "weights": [float(m) for m in self.weights]
        }

    def __repr__(self):
        template = "{self.__class__.__name__}({self.shapes}, {self.scales}, {self.weights})"
        return template.format(self=self)

    def _logpdf(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Computes the log-space density for `X` using the `k`th
        component of the mixture
        """
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
    def fit(cls, X: np.ndarray, n_components: int, maxiter: int=100, tol: float=1e-5, deterministic: bool=True) -> 'GammaMixtureBase':
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

    def estimate(self, X: np.ndarray, maxiter=100, tol=1e-5):
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
    prior: MixtureBase

    def __init__(self, mus, sigmas, prior, weights):
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.prior = prior
        self.weights = np.array(weights)
        self.n_components = len(weights)

    def to_json(self) -> Dict:
        state = super().to_json()
        state['prior'] = self.prior.to_json()
        state['prior_type'] = self.prior.__class__.__name__
        return state

    @classmethod
    def from_json(cls, state: Dict) -> 'GaussianMixtureWithPriorComponent':
        # TODO: Need to derive the type object of the prior from its name
        # prior_type_name = state['prior_type']
        prior = GammaMixtureBase.from_json(state['prior'])
        return cls(state['mus'], state['sigmas'], prior, state['weights'])

    def _logpdf(self, X, k):
        if k == self.n_components - 1:
            return np.log(np.exp(self.prior.logpdf(X, weighted=False)).dot(self.prior.weights))
        else:
            return super(GaussianMixtureWithPriorComponent, self)._logpdf(X, k)

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int, prior: MixtureBase, maxiter=1000, tol=1e-5, deterministic=True) -> 'GaussianMixtureWithPriorComponent':
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

    def estimate(self, X: np.ndarray, maxiter=1000, tol=1e-5):
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
    def fit(cls, X, n_components, prior, maxiter=1000, tol=1e-5, deterministic=True) -> 'TruncatedGaussianMixtureWithPriorComponent':
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


class PredictorBase(Protocol):
    def predict(self, value: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        ...


class KDModel(PredictorBase):
    positive_probabilities: array
    negative_probabilities: array
    values: array

    positive_prior: float

    positive_kernel_fit: stats.gaussian_kde
    negative_kernel_fit: stats.gaussian_kde

    def __init__(self,
                 positive_prior: float=None,
                 positive_probabilities: array=None,
                 negative_probabilities: array=None,
                 values: array=None):
        if positive_probabilities is None:
            positive_probabilities = array('d')
        if negative_probabilities is None:
            negative_probabilities = array('d')
        if values is None:
            values = array('d')
        if positive_prior is None:
            positive_prior = 0.5

        self.positive_probabilities = positive_probabilities
        self.negative_probabilities = negative_probabilities
        self.values = values

        self.positive_prior = positive_prior

        self.positive_kernel_fit = None
        self.negative_kernel_fit = None

    def add(self, prob: float, val: float):
        self.positive_probabilities.append(prob)
        self.negative_probabilities.append(1 - prob)
        self.values.append(val)

    def fit(self, maxiter: int=10):
        lastprobsum = self.update()

        i = 0
        while i < maxiter:
            probsum = self.update()
            i += 1
            if abs(probsum - lastprobsum) < 0.001:
                break
            lastprobsum = probsum

    def update_fits(self):
        self.positive_kernel_fit = GaussianKDE(
            self.values, weights=self.positive_probabilities)
        self.negative_kernel_fit = GaussianKDE(
            self.values, weights=self.negative_probabilities)

    def update(self):
        self.update_fits()
        probs = self.predict(self.values)
        count = len(probs)
        probsum = np.sum(probs)
        self.positive_prior = probsum / count
        self.positive_probabilities = probs
        self.negative_probabilities = 1 - probs
        return probsum

    def predict(self, value: float) -> float:
        if self.positive_kernel_fit is None:
            raise TypeError("Cannot predict with a model that has not been fit yet")

        p = self.predict_positive(value)
        n = self.predict_negative(value)

        pos_prior = self.positive_prior
        neg_prior = 1 - pos_prior
        return (p * pos_prior) / ((p * pos_prior) + (n * neg_prior))

    def predict_positive(self, value: float) -> float:
        return self.positive_kernel_fit.pdf(value)

    def predict_negative(self, value: float) -> float:
        return self.negative_kernel_fit.pdf(value)


class GaussianKDE(stats.gaussian_kde):
    '''To control overriding methods, use a derived class'''
    pass


KDE_USE_THREADS = True
try:
    from glycresoft._c.structure.probability import evaluate_gaussian_kde
    GaussianKDE.evaluate = evaluate_gaussian_kde
except ImportError as err:
    print(err)
    KDE_USE_THREADS = False


class MultiKDModel(PredictorBase):
    models: List[KDModel]
    values: List[array]
    positive_prior: float

    use_threads: bool
    thread_pool: Optional[ThreadPoolExecutor]

    def __init__(self,
                 positive_prior: float=None,
                 models: List[KDModel]=None,
                 values: array=None,
                 use_threads: bool=KDE_USE_THREADS):
        if positive_prior is None:
            positive_prior = 0.5
        if models is None:
            models = []
        if values is None:
            values = [array('d') for m in models]
        self.models = models
        self.positive_prior = positive_prior
        self.values = values
        self.use_threads = use_threads
        self.thread_pool = None
        if self.use_threads:
            self.thread_pool = ThreadPoolExecutor()

    def __getstate__(self):
        state = {
            'positive_prior': self.positive_prior,
            'models': self.models,
            'values': self.values,
        }
        return state

    def __setstate__(self, state):
        self.positive_prior = state['positive_prior']
        self.models = state['models']
        self.values = state['values']

    def __reduce__(self):
        return self.__class__, (), self.__getstate__()

    def add_model(self, model: KDModel):
        self.models.append(model)
        self.values.append(model.values)

    def add(self, prob: float, values: List[float]):
        for val, mod, valn in zip(values, self.models, self.values):
            mod.add(prob, val)
            valn.append(val)

    def update_fits(self):
        for model in self.models:
            model.update_fits()

    def close_thread_pool(self):
        if self.use_threads:
            if self.thread_pool is not None:
                self.thread_pool.shutdown()
            self.thread_pool =  None
            self.use_threads = False

    def create_thread_pool(self):
        self.use_threads = True
        self.thread_pool = ThreadPoolExecutor()

    def update(self):
        self.update_fits()
        probs = self.predict(self.values)
        count = len(probs)
        probsum = np.sum(probs)
        self.positive_prior = probsum / count
        for mod in self.models:
            mod.positive_probabilities = probs
            mod.negative_probabilities = 1 - probs
        return probsum

    def predict_positive(self, values: List[float]) -> float:
        acc = 0
        if self.use_threads:
            jobs = []
            for mod, val in zip(self.models, values):
                jobs.append(self.thread_pool.submit(mod.positive_kernel_fit.evaluate, val))

            for job in jobs:
                acc += np.log(job.result())
        else:
            for mod, val in zip(self.models, values):
                acc += np.log(mod.positive_kernel_fit.evaluate(val))
        return np.exp(acc)

    def predict_negative(self, values: List[float]) -> float:
        acc = 0
        if self.use_threads:
            jobs = []
            for mod, val in zip(self.models, values):
                jobs.append(self.thread_pool.submit(
                    mod.negative_kernel_fit.evaluate, val))

            for job in jobs:
                acc += np.log(job.result())
        else:
            for mod, val in zip(self.models, values):
                acc += np.log(mod.negative_kernel_fit.evaluate(val))
        return np.exp(acc)

    def predict(self, values: List[float]) -> float:
        if self.use_threads:
            p_fut = self.thread_pool.submit(self.predict_positive, values)
            n_fut = self.thread_pool.submit(self.predict_negative, values)
            p = p_fut.result()
            n = n_fut.result()
        else:
            p = self.predict_positive(values)
            n = self.predict_negative(values)

        pos_prior = self.positive_prior
        neg_prior = 1 - pos_prior
        return (p * pos_prior) / ((p * pos_prior) + (n * neg_prior))

    def fit(self, maxiter: int=10, convergence: float=0.001):
        lastprobsum = self.update()
        delta = float('inf')
        i = 1
        while i < maxiter:
            probsum = self.update()
            i += 1
            delta = abs(probsum - lastprobsum)
            if delta < convergence:
                break
            lastprobsum = probsum
        return i, delta
