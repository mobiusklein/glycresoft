from collections import namedtuple, defaultdict
import random

import numpy as np


from matplotlib import pyplot as plt

from glycan_profiling.scoring.base import symbolic_composition, is_sialylated, degree_of_sialylation


WLSSolution = namedtuple("WLSSolution", [
    'yhat', 'parameters', 'data', 'weights', 'residuals',
    'projection_matrix', 'rss', 'press', 'R2'])


def prepare_arrays_for_linear_fit(x, y, w=None):
    X = np.vstack((np.ones(len(x)), np.array(x))).T
    Y = np.array(y)
    if w is None:
        W = np.eye(Y.shape[0])
    else:
        W = np.array(w)
    return X, Y, W


def weighted_linear_regression_fit(x, y, w=None, prepare=False):
    if prepare:
        x, y, w = prepare_arrays_for_linear_fit(x, y, w)
    elif w is None:
        w = np.eye(y.shape[0])
    A = np.linalg.pinv(x.T.dot(w).dot(x)).dot(x.T.dot(w))
    B = A.dot(y)
    H = x.dot(A)
    yhat = x.dot(B)
    residuals = (y - yhat)
    leave_one_out_error = residuals / (1 - np.diag(H))
    press = (np.diag(w) * leave_one_out_error * leave_one_out_error).sum()
    rss = (np.diag(w) * residuals * residuals).sum()
    tss = (y - y.mean())
    tss = (np.diag(w) * tss * tss).sum()
    return WLSSolution(
        yhat, B, (x, y), w, residuals, H,
        rss, press, 1 - (rss / (tss)))


def ransac(x, y, w=None, max_trials=100):
    '''
    RANSAC Regression, inspired heavily by sklearn's
    much more complex implementation
    '''
    X = x
    residual_threshold = np.median(np.abs(y - np.median(y)))

    if w is None:
        w = np.eye(y)

    def loss(y_true, y_pred):
        return np.abs(y_true - y_pred)

    n_trials = 0
    n_samples = X.shape[0]
    min_samples = X.shape[1] * 5
    if min_samples > X.shape[0]:
        min_samples = X.shape[1] + 1

    if min_samples > X.shape[0]:
        return weighted_linear_regression_fit(X, y, w)

    sample_indices = np.arange(n_samples)

    rng = random.Random(1)

    n_inliers_best = 1
    score_best = -np.inf
    X_inlier_best = None
    y_inlier_best = None

    while n_trials < max_trials:
        n_trials += 1
        subset_ix = rng.sample(sample_indices, min_samples)
        X_subset = X[subset_ix]
        y_subset = y[subset_ix]
        w_subset = np.diag(np.diag(w)[subset_ix])

        # fit parameters on random subset of the data
        fit = weighted_linear_regression_fit(X_subset, y_subset, w_subset)

        # compute goodness of fit for the fitted parameters with
        # the full dataset
        yhat = np.dot(X, fit.parameters)
        residuals_subset = loss(y, yhat)

        # locate inliers based on residual threshold
        inlier_subset_mask = residuals_subset < residual_threshold
        n_inliers_subset = inlier_subset_mask.sum()

        # determine the quality of the fitted parameters for
        # the inliers using R2
        inlier_subset_ix = sample_indices[inlier_subset_mask]
        X_inlier_subset = X[inlier_subset_ix]
        y_inlier_subset = y[inlier_subset_ix]

        yhat_inlier_subset = X_inlier_subset.dot(fit.parameters)
        rss = np.square(y_inlier_subset - yhat_inlier_subset).sum()
        tss = np.square(y_inlier_subset - y_inlier_subset.mean()).sum()

        score_subset = 1 - (rss / tss)

        # If the number of inliers chosen hasn't improved and the score hasn't
        # improved, don't update the current best
        if n_inliers_subset < n_inliers_best and score_subset < score_best:
            continue

        score_best = score_subset
        X_inlier_best = X_inlier_subset
        y_inlier_best = y_inlier_subset

    # fit the final best inlier set for the final parameters
    return weighted_linear_regression_fit(X_inlier_best, y_inlier_best)


def prediction_interval(solution, x0, y0):
    pred_variance = solution.rss / (len(solution.residuals) - len(solution.parameters))
    X = solution.data[0]
    w = solution.weights
    xtx_inv = np.linalg.pinv(X.T.dot(w).dot(X))
    h = x0.dot(xtx_inv).dot(x0.T)
    if not np.isscalar(h):
        h = np.diag(h)
    forecast_variance = 2 * pred_variance * h
    Z = 1.960
    width = Z * np.sqrt(forecast_variance)
    return np.stack([y0 - width, y0 + width])


class ElutionTimeFitter(object):
    def __init__(self, chromatograms):
        self.chromatograms = chromatograms
        self.neutral_mass_array = np.array([
            x.weighted_neutral_mass for x in chromatograms
        ])
        self.data = self._prepare_data_matrix(self.neutral_mass_array)

        self.apex_time_array = np.array([
            x.apex_time for x in chromatograms
        ])

        self.weight_matrix = self.build_weight_matrix()

        self.parameters = None
        self.residuals = None
        self.estimate = None

    def build_weight_matrix(self):
        return np.eye(len(self.chromatograms))

    def _prepare_data_vector(self, chromatogram):
        return np.array([1,
                         chromatogram.weighted_neutral_mass,
                         # chromatogram.weighted_neutral_mass ** 2
                         ])

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
            # np.square(np.array(mass_array))
        )).T

    def _fit(self, resample=True):
        if resample:
            solution = ransac(self.data, self.apex_time_array, self.weight_matrix)
        else:
            solution = weighted_linear_regression_fit(
                self.data, self.apex_time_array, self.weight_matrix)
        return solution

    def fit(self, bootstrap=True):
        solution = self._fit(bootstrap)
        self.estimate = solution.yhat
        self.residuals = solution.residuals
        self.parameters = solution.parameters
        self.projection_matrix = solution.projection_matrix
        self.rss = solution.rss
        self.solution = solution

    def predict(self, chromatogram):
        return self._predict(self._prepare_data_vector(chromatogram))

    def _predict(self, x):
        return x.dot(self.parameters)

    def predict_ci(self, chromatogram):
        x = self._prepare_data_vector(chromatogram)
        return self._predict_ci(x)

    def _predict_ci(self, x):
        y = self._predict(x)
        return prediction_interval(self.solution, x, y)

    def score(self, chromatogram):
        apex = self.predict(chromatogram)
        score = (1 - abs(apex - chromatogram.apex_time) / max(
                 apex, chromatogram.apex_time)) / 1.5
        return score - 1e-6

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.scatter(self.neutral_mass_array, self.apex_time_array, label='Observed')
        theoretical_mass = np.linspace(
            max(self.neutral_mass_array.min() - 200, 0),
            self.neutral_mass_array.max() + 200, 400)
        X = self._prepare_data_matrix(theoretical_mass)
        Y = X.dot(self.parameters)
        ax.plot(theoretical_mass, Y, linestyle='--', label='Trend Line')
        pred_interval = self._predict_ci(X)
        ax.fill_between(
            theoretical_mass, pred_interval[0, :], pred_interval[1, :],
            alpha=0.4, label='Prediction Interval')
        return ax

    def clone(self):
        return self.__class__(self.chromatograms)


class ScoreWeightedElutionTimeFitter(ElutionTimeFitter):
    def build_weight_matrix(self):
        return np.eye(len(self.chromatograms)) * [
            x.score for x in self.chromatograms
        ]


class AbundanceWeightedElutionTimeFitter(ElutionTimeFitter):
    def build_weight_matrix(self):
        return np.eye(len(self.chromatograms)) * [
            x.total_signal for x in self.chromatograms
        ]


def is_high_mannose(composition):
    return (composition['HexNAc'] == 2 and composition['Hex'] > 3 and
            not is_sialylated(composition))


class PartitioningElutionTimeFitter(ElutionTimeFitter):

    def __init__(self, chromatograms, fitter_cls=ElutionTimeFitter):
        self.fitter_cls = fitter_cls
        self.subfits = dict()
        self.chromatograms = chromatograms
        self.partitions = self.partition_chromatograms(chromatograms)

    def label(self, chromatogram):
        if chromatogram.composition:
            composition = symbolic_composition(chromatogram)
            if is_high_mannose(composition):
                return 'high_mannose'
            else:
                return 'other'
        else:
            return 'other'

    def fit(self):
        for group, members in self.partitions.items():
            subfit = self.fitter_cls(members)
            subfit.fit()
            self.subfits[group] = subfit

    def partition_chromatograms(self, chromatograms):
        partitions = defaultdict(list)
        for chromatogram in chromatograms:
            partitions[self.label(chromatogram)].append(chromatogram)
        return partitions

    def predict(self, chromatogram):
        label = self.label(chromatogram)
        fit = self.subfits[label]
        return fit.predict(chromatogram)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        for group, fit in self.subfits.items():
            label = ' '.join(group.split("_")).title()
            r2 = fit.solution.R2
            art = ax.scatter(fit.neutral_mass_array, fit.apex_time_array,
                             label=r"%s (${\bar R^2}$: %0.3f)" % (label, r2),
                             alpha=0.8)
            color = art.get_facecolor()[0]
            theoretical_mass = np.linspace(
                max(fit.neutral_mass_array.min() - 200, 0),
                fit.neutral_mass_array.max() + 200, 400)
            X = fit._prepare_data_matrix(theoretical_mass)
            Y = X.dot(fit.parameters)
            ax.plot(theoretical_mass, Y, linestyle='--', color=color)
            pred_interval = fit._predict_ci(X)
            ax.fill_between(
                theoretical_mass, pred_interval[0, :], pred_interval[1, :],
                alpha=0.4, color=color)
        ax.set_xlabel("Neutral Mass", fontsize=16)
        ax.set_ylabel("Elution Apex Time\n(Minutes)", fontsize=16)
        return ax
