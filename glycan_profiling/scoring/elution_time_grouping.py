from collections import namedtuple, defaultdict
import random

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from glycopeptidepy import PeptideSequence
from glypy.utils import make_counter

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from glycan_profiling.scoring.base import (
    ScoringFeatureBase,
    symbolic_composition,
    is_sialylated)


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
    w_inlier_best = None

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
        w_inlier_subset = np.diag(np.diag(w)[inlier_subset_ix])
        # w_inlier_best = 1

        yhat_inlier_subset = X_inlier_subset.dot(fit.parameters)
        rss = (w_inlier_subset * np.square(
            y_inlier_subset - yhat_inlier_subset)).sum()
        tss = (w_inlier_subset * np.square(
            y_inlier_subset - y_inlier_subset.mean())).sum()

        score_subset = 1 - (rss / tss)

        # If the number of inliers chosen hasn't improved and the score hasn't
        # improved, don't update the current best
        if n_inliers_subset < n_inliers_best and score_subset < score_best:
            continue

        score_best = score_subset
        X_inlier_best = X_inlier_subset
        y_inlier_best = y_inlier_subset
        w_inlier_best = w_inlier_subset

    # fit the final best inlier set for the final parameters
    return weighted_linear_regression_fit(X_inlier_best, y_inlier_best, w_inlier_best)


def prediction_interval(solution, x0, y0, alpha=0.05):
    n = len(solution.residuals)
    k = len(solution.parameters)
    df = n - k
    sigma2 = solution.rss / df
    X = solution.data[0]
    w = solution.weights

    xtx_inv = np.linalg.pinv(X.T.dot(w).dot(X))
    h = x0.dot(xtx_inv).dot(x0.T)
    if not np.isscalar(h):
        h = np.diag(h)
    error_of_prediction = np.sqrt(sigma2 * (1 + h))

    t = stats.t.isf(alpha / 2., df)
    width = t * error_of_prediction
    return np.stack([y0 - width, y0 + width])


class ChromatogramProxy(object):
    def __init__(self, weighted_neutral_mass, apex_time, total_signal, glycan_composition, obj=None):
        self.weighted_neutral_mass = weighted_neutral_mass
        self.apex_time = apex_time
        self.total_signal = total_signal
        self.glycan_composition = glycan_composition
        self.obj = obj

    def __repr__(self):
        return "%s(%f, %f, %f, %s)" % (
            self.__class__.__name__,
            self.weighted_neutral_mass, self.apex_time, self.total_signal, self.glycan_composition)

    @classmethod
    def from_obj(cls, obj):
        inst = cls(
            obj.weighted_neutral_mass, obj.apex_time, obj.total_signal,
            obj.glycan_composition, obj)
        return inst

    def get_chromatogram(self):
        return self.obj.get_chromatogram()


class GlycopeptideChromatogramProxy(ChromatogramProxy):
    @property
    def structure(self):
        gp = PeptideSequence(str(self.obj.structure))
        return gp


class ElutionTimeFitter(ScoringFeatureBase):
    feature_type = 'elution_time'

    def __init__(self, chromatograms, scale=1):
        self.chromatograms = chromatograms
        self.neutral_mass_array = np.array([
            x.weighted_neutral_mass for x in chromatograms
        ])
        self.data = self._prepare_data_matrix(self.neutral_mass_array)

        self.apex_time_array = np.array([
            self._get_apex_time(x) for x in chromatograms
        ])

        self.weight_matrix = self.build_weight_matrix()

        self.parameters = None
        self.residuals = None
        self.estimate = None
        self.scale = scale

    def _get_apex_time(self, chromatogram):
        try:
            x, y = chromatogram.as_arrays()
            y = gaussian_filter1d(y, 1)
            return x[np.argmax(y)]
        except AttributeError:
            return chromatogram.apex_time

    def build_weight_matrix(self):
        return np.eye(len(self.chromatograms))

    def _prepare_data_vector(self, chromatogram):
        return np.array([1,
                         chromatogram.weighted_neutral_mass,
                        ])

    def feature_names(self):
        return ['intercept', 'mass']

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
        )).T

    def _fit(self, resample=True):
        if resample:
            solution = ransac(self.data, self.apex_time_array, self.weight_matrix)
            alt = weighted_linear_regression_fit(self.data, self.apex_time_array, self.weight_matrix)
            if alt.R2 > solution.R2:
                return alt
            return solution
        else:
            solution = weighted_linear_regression_fit(
                self.data, self.apex_time_array, self.weight_matrix)
        return solution

    def fit(self, resample=True):
        solution = self._fit(resample=resample)
        self.estimate = solution.yhat
        self.residuals = solution.residuals
        self.parameters = solution.parameters
        self.projection_matrix = solution.projection_matrix
        self.solution = solution
        return self

    @property
    def rss(self):
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss = (np.diag(w) * residuals * residuals).sum()
        return rss

    @property
    def mse(self):
        return self.rss / (len(self.apex_time_array) - len(self.parameters) - 1.0)

    def parameter_significance(self):
        XtWX_inv = np.linalg.pinv((self.data.T.dot(self.weight_matrix).dot(self.data)))
        # With unknown variance, use the mean squared error estimate
        sigma_params = np.sqrt(np.diag(self.mse * XtWX_inv))
        degrees_of_freedom = len(self.apex_time_array) - len(self.parameters) - 1.0
        # interval = stats.t.interval(1 - alpha / 2.0, degrees_of_freedom)
        t_score = np.abs(self.parameters) / sigma_params
        p_value = stats.t.sf(t_score, degrees_of_freedom) * 2
        return p_value

    def parameter_confidence_interval(self, alpha=0.05):
        X = self.data
        sigma_params = np.sqrt(
            np.diag((self.mse) * np.linalg.pinv(
                X.T.dot(self.weight_matrix).dot(X))))
        degrees_of_freedom = len(self.apex_time_array) - \
            len(self.parameters) - 1
        iv = stats.t.interval((1 - alpha) / 2., degrees_of_freedom)
        iv = np.array(iv) * sigma_params.reshape((-1, 1))
        return np.array(self.parameters).reshape((-1, 1)) + iv

    def R2(self, adjust=False):
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss = (np.diag(w) * residuals * residuals).sum()
        tss = (y - y.mean())
        tss = (np.diag(w) * tss * tss).sum()
        n = len(y)
        k = len(self.parameters)
        if adjust:
            adjustment_factor = (n - 1.0) / float(n - k - 1.0)
        else:
            adjustment_factor = 1.0
        R2 = (1 - adjustment_factor * (rss / tss))
        return R2

    def predict(self, chromatogram):
        return self._predict(self._prepare_data_vector(chromatogram))

    def _predict(self, x):
        return x.dot(self.parameters)

    def predict_interval(self, chromatogram):
        x = self._prepare_data_vector(chromatogram)
        return self._predict_interval(x)

    def _predict_interval(self, x):
        y = self._predict(x)
        return prediction_interval(self.solution, x, y)

    def score(self, chromatogram):
        apex = self.predict(chromatogram)
        # Use heavier tails (scale 2) to be more tolerant of larger chromatographic
        # errors.
        # The survival function's maximum value is 0.5, so double this to map the
        # range of values to be (0, 1)
        score = stats.t.sf(
            abs(apex - self._get_apex_time(chromatogram)),
            df=max(len(self.chromatograms) - len(self.parameters), 1), scale=self.scale) * 2
        return max((score - 1e-3), 1e-3)

    def plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        ax.scatter(self.neutral_mass_array, self.apex_time_array, label='Observed')
        theoretical_mass = np.linspace(
            max(self.neutral_mass_array.min() - 200, 0),
            self.neutral_mass_array.max() + 200, 400)
        X = self._prepare_data_matrix(theoretical_mass)
        Y = X.dot(self.parameters)
        ax.plot(theoretical_mass, Y, linestyle='--', label='Trend Line')
        pred_interval = self._predict_interval(X)
        ax.fill_between(
            theoretical_mass, pred_interval[0, :], pred_interval[1, :],
            alpha=0.4, label='Prediction Interval')
        return ax

    def clone(self):
        return self.__class__(self.chromatograms)


class AbundanceWeightedElutionTimeFitter(ElutionTimeFitter):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            np.log10(x.total_signal) for x in self.chromatograms
        ]
        W /= W.max()
        return W


class FactorElutionTimeFitter(ElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self.factors = list(factors)
        super(FactorElutionTimeFitter, self).__init__(chromatograms, scale=scale)

    def feature_names(self):
        return ['intercept'] + self.factors

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
        ) + tuple(
            np.array([c.glycan_composition[f] for c in self.chromatograms])
            for f in self.factors)
        ).T

    def _prepare_data_vector(self, chromatogram):
        return np.array(
            [1,
             ] + [
                chromatogram.glycan_composition[f] for f in self.factors])

    def plot(self, ax=None, include_intervals=True):
        from glycan_profiling.plotting.colors import ColorMapper
        if ax is None:
            _fig, ax = plt.subplots(1)
        colorer = ColorMapper()
        factors = self.factors
        column_offset = 1
        distinct_combinations = set()
        partitions = defaultdict(list)
        for i, row in enumerate(self.data):
            key = tuple(row[column_offset:])
            distinct_combinations.add(key)
            partitions[key].append((self.neutral_mass_array[i], self.apex_time_array[i]))

        theoretical_mass = np.linspace(
            max(self.neutral_mass_array.min() - 200, 0),
            self.neutral_mass_array.max() + 200, 400)
        for combination in distinct_combinations:
            members = partitions[combination]
            ox, oy = zip(*members)
            v = np.ones_like(theoretical_mass)
            factor_partition = [v * f for f in combination]
            label_part = ','.join(["%s:%d" % (fl, fv) for fl, fv in zip(
                factors, combination)])
            color = colorer[combination]
            ax.scatter(ox, oy, label=label_part, color=color)
            X = np.vstack([
                np.ones_like(theoretical_mass),
                # theoretical_mass,
            ] + factor_partition).T
            Y = X.dot(self.parameters)
            ax.plot(
                theoretical_mass, Y, linestyle='--', color=color)
            if include_intervals:
                pred_interval = self._predict_interval(X)
                ax.fill_between(
                    theoretical_mass, pred_interval[0, :], pred_interval[1, :],
                    alpha=0.4, color=color)

        return ax

    def clone(self):
        return self.__class__(self.chromatograms, factors=self.factors, scale=self.scale)


class AbundanceWeightedFactorElutionTimeFitter(FactorElutionTimeFitter):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            (x.total_signal) for x in self.chromatograms
        ]
        W /= W.max()
        return W


class PeptideFactorElutionTimeFitter(FactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self._peptide_to_indicator = defaultdict(make_counter(0))
        # Ensure that _peptide_to_indicator is properly initialized
        for obs in chromatograms:
            _ = self._peptide_to_indicator[self._get_peptide_key(obs)]
        super(PeptideFactorElutionTimeFitter, self).__init__(chromatograms, list(factors), scale)

    def _get_peptide_key(self, chromatogram):
        return PeptideSequence(str(chromatogram.structure)).deglycosylate()

    def _prepare_data_matrix(self, mass_array):
        p = len(self._peptide_to_indicator)
        n = len(self.chromatograms)
        peptides = np.zeros((p, n))
        indicator = dict(self._peptide_to_indicator)
        for i, c in enumerate(self.chromatograms):
            try:
                j = indicator[self._get_peptide_key(c)]
                peptides[j, i] = 1
            except KeyError:
                pass
        # Omit the intercept, so that all peptide levels are used without inducing linear dependence.
        return np.vstack((
            peptides,
        ) + tuple(
            np.array([c.glycan_composition[f] for c in self.chromatograms])
            for f in self.factors)
        ).T

    def feature_names(self):
        names = []
        peptides = [None] * len(self._peptide_to_indicator)
        for key, value in self._peptide_to_indicator.items():
            peptides[value] = key
        names.extend(peptides)
        names.extend(self.factors)
        return names

    def _prepare_data_vector(self, chromatogram):
        p = len(self._peptide_to_indicator)
        peptides = [0 for i in range(p)]
        indicator = dict(self._peptide_to_indicator)
        try:
            key_index = self._get_peptide_key(chromatogram)
            peptides[indicator[key_index]] = 1
        except KeyError:
            pass
        return np.array(
            peptides + [chromatogram.glycan_composition[f] for f in self.factors])


class AbundanceWeightedPeptideFactorElutionTimeFitter(PeptideFactorElutionTimeFitter):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            (x.total_signal) for x in self.chromatograms
        ]
        W /= W.max()
        return W


class ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter(AbundanceWeightedPeptideFactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self._replicate_to_indicator = defaultdict(make_counter(0))
        # Ensure that _replicate_to_indicator is properly initialized
        for obs in chromatograms:
            _ = self._replicate_to_indicator[self._get_replicate_key(obs)]
        super(ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter, self).__init__(
            chromatograms, list(factors), scale)

    def _get_replicate_key(self, chromatogram):
        return getattr(chromatogram, 'replicate_id')

    def _prepare_data_matrix(self, mass_array):
        design_matrix = super(
            ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter,
            self)._prepare_data_matrix(mass_array)
        p = len(self._replicate_to_indicator)
        n = len(self.chromatograms)
        replicate_matrix = np.zeros((p, n))
        indicator = dict(self._replicate_to_indicator)
        for i, c in enumerate(self.chromatograms):
            try:
                # Here, if one of the levels is not omitted, the matrix will be linearly dependent.
                # So drop the 0th factor level.
                j = indicator[self._get_replicate_key(c)]
                if j != 0:
                    replicate_matrix[j, i] = 1
            except KeyError:
                pass
        return np.hstack((replicate_matrix.T, design_matrix))

    def feature_names(self):
        names = []
        replicates = [None] * len(self._replicate_to_indicator)
        for key, value in self._replicate_to_indicator.items():
            replicates[value] = key
        names.extend(replicates)
        peptides = [None] * len(self._peptide_to_indicator)
        for key, value in self._peptide_to_indicator.items():
            peptides[value] = key
        names.extend(peptides)
        names.extend(self.factors)
        return names

    def _prepare_data_vector(self, chromatogram):
        data_vector = super(
            ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter,
            self)._prepare_data_vector(chromatogram)
        p = len(self._replicate_to_indicator)
        replicates = [0 for i in range(p)]
        indicator = dict(self._replicate_to_indicator)
        try:
            # Here, if one of the levels is not omitted, the matrix will be linearly dependent.
            # So drop the 0th factor level.
            j = self._get_replicate_key(chromatogram)
            if j != 0:
                replicates[indicator[j]] = 1
        except KeyError:
            pass
        return np.hstack((replicates, data_vector))


def is_high_mannose(composition):
    return (composition['HexNAc'] == 2 and composition['Hex'] > 3 and
            not is_sialylated(composition))


class PartitioningElutionTimeFitter(ElutionTimeFitter):

    def __init__(self, chromatograms, scale=1, fitter_cls=ElutionTimeFitter):
        self.fitter_cls = fitter_cls
        self.subfits = dict()
        self.chromatograms = chromatograms
        self.scale = scale
        self.partitions = self.partition_chromatograms(chromatograms)

    def label(self, chromatogram):
        if chromatogram.composition:
            composition = symbolic_composition(chromatogram)
            if is_high_mannose(composition):
                return 'high_mannose'
            else:
                return 'other'
        else:
            return 'unassigned'

    def fit(self, resample=True):
        for group, members in self.partitions.items():
            subfit = self.fitter_cls(members, scale=self.scale)
            subfit.fit(resample=resample)
            self.subfits[group] = subfit
        return self

    def partition_chromatograms(self, chromatograms):
        partitions = defaultdict(list)
        for chromatogram in chromatograms:
            label = self.label(chromatogram)
            if label != "unassigned":
                partitions[label].append(chromatogram)
        return partitions

    def predict(self, chromatogram):
        label = self.label(chromatogram)
        if label != 'unassigned':
            fit = self.subfits[label]
            return fit.predict(chromatogram)
        else:
            group, closest = self._find_best_fit_for_unassigned(chromatogram)
            return closest

    def predict_ci(self, chromatogram):
        label = self.label(chromatogram)
        if label != 'unassigned':
            fit = self.subfits[label]
            return fit.predict_ci(chromatogram)
        else:
            group, closest = self._find_best_fit_for_unassigned(chromatogram)
            subfit = self.subfits[group]
            return subfit.predict_ci(chromatogram)

    def _find_best_fit_for_unassigned(self, chromatogram):
        closest = 0
        distance = float('inf')
        closest_group = None
        for group, subfit in self.subfits.items():
            predicted = subfit.predict(chromatogram)
            delta = abs(predicted - chromatogram.apex_time)
            if delta < distance:
                closest = predicted
                distance = delta
                closest_group = group
        return closest_group, closest

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
            pred_interval = fit._predict_interval(X)
            ax.fill_between(
                theoretical_mass, pred_interval[0, :], pred_interval[1, :],
                alpha=0.4, color=color)
        ax.set_xlabel("Neutral Mass", fontsize=16)
        ax.set_ylabel("Elution Apex Time\n(Minutes)", fontsize=16)
        return ax

    def R2(self, adjust=False):
        total = 0
        weights = 0
        for group, subfit in self.subfits.items():
            weight = subfit.weight_matrix.sum() * 100
            total += subfit.R2(adjust) * weight
            weights += weight
        return total / weights


class ElutionTimeModel(ScoringFeatureBase):
    feature_type = 'elution_time'

    def __init__(self, fit=None):
        self.fit = fit

    def configure(self, analysis_data):
        if self.fit is None:
            matches = analysis_data['matches']
            fitter = PartitioningElutionTimeFitter(matches)
            fitter.fit()
            self.fit = fitter

    def score(self, chromatogram, *args, **kwargs):
        if self.fit is not None:
            return self.fit.score(chromatogram)
        else:
            return 0.5


CalibrationPoint = namedtuple("CalibrationPoint", (
    "reference_index", "reference_point_rt", "reference_point_rt_pred",
    # rrt = Relative Retention Time, prrt = Predicted Relative Retention Time
    "rrt", "prrt", "residuals", 'weight'))


class RecalibratingPredictor(object):
    def __init__(self, training_examples, testing_examples, model, scale=1.0, dilation=1.0, weighted=True):
        if training_examples is None:
            training_examples = []
        self.training_examples = np.array(training_examples)
        self.testing_examples = np.array(testing_examples)
        self.model = model
        self.scale = scale
        self.dilation = dilation
        self.configurations = dict()
        self._fit()
        self.apex_time_array = np.array([self.model._get_apex_time(c) for c in testing_examples])
        if weighted:
            self.weight_array = np.array([np.log10(c.total_signal) for c in testing_examples])
            self.weight_array /= self.weight_array.max()
        else:
            self.weight_array = np.ones_like(self.apex_time_array)
        self.weighted = weighted
        self.weight_array /= self.weight_array.max()
        self.predicted_apex_time_array = self._predict()
        self.score_array = self._score()

    def _adapt_dilate_fit(self, reference_point, dilation):
        parameters = np.hstack([self.model.parameters[0], dilation * self.model.parameters[1:]])
        predicted_reference_point_rt = self.model._prepare_data_vector(reference_point).dot(parameters)
        reference_point_rt = self.model._get_apex_time(reference_point)
        resid = []
        relative_retention_time = []
        predicted_relative_retention_time = []
        for ex in self.testing_examples:
            rrt = self.model._get_apex_time(ex) - reference_point_rt
            prrt = self.model._prepare_data_vector(ex).dot(parameters) - predicted_reference_point_rt
            relative_retention_time.append(rrt)
            predicted_relative_retention_time.append(prrt)
            resid.append(prrt - rrt)
        return (reference_point_rt, predicted_reference_point_rt,
                relative_retention_time, predicted_relative_retention_time, resid)

    def _predict_delta_single(self, test_point, reference_point, dilation=None):
        if dilation is None:
            dilation = self.dilation
        parameters = np.hstack([self.model.parameters[0], dilation * self.model.parameters[1:]])
        predicted_reference_point_rt = self.model._prepare_data_vector(reference_point).dot(parameters)
        reference_point_rt = self.model._get_apex_time(reference_point)
        rrt = self.model._get_apex_time(test_point) - reference_point_rt
        prrt = self.model._prepare_data_vector(test_point).dot(parameters) - predicted_reference_point_rt
        return prrt - rrt

    def predict_delta_single(self, test_point, dilation=None):
        if dilation is None:
            dilation = self.dilation
        delta = []
        weight = []
        for i, reference_point in enumerate(self.testing_examples):
            delta.append(self._predict_delta_single(test_point, reference_point, dilation))
            weight.append(np.log10(reference_point.total_signal))
        return np.dot(delta, weight) / np.sum(weight), np.std(delta)

    def score_single(self, test_point):
        delta, sd = (self.predict_delta_single(test_point))
        delta = abs(delta)
        score = stats.t.sf(
            delta,
            df=self._df(), scale=self.scale) * 2
        score -= 1e-3
        if score < 1e-3:
            score = 1e-3
        return score

    def _fit(self):
        for i, training_reference_point in enumerate(self.training_examples):
            reference_point = [
                c for c in self.testing_examples
                if (c.glycan_composition == training_reference_point.glycan_composition)]
            if not reference_point:
                continue
            reference_point = max(reference_point, key=lambda x: x.total_signal)
            dilation = self.dilation
            (reference_point_rt, predicted_reference_point_rt,
             relative_retention_time,
             predicted_relative_retention_time, resid) = self._adapt_dilate_fit(reference_point, dilation)
            self.configurations[i] = CalibrationPoint(
                i, reference_point_rt, predicted_reference_point_rt,
                np.array(relative_retention_time), np.array(predicted_relative_retention_time),
                np.array(resid), np.log10(reference_point.total_signal)
            )
        if not self.configurations:
            for i, reference_point in enumerate(self.testing_examples):
                dilation = self.dilation
                (reference_point_rt, predicted_reference_point_rt,
                 relative_retention_time,
                 predicted_relative_retention_time, resid) = self._adapt_dilate_fit(reference_point, dilation)
                self.configurations[-i] = CalibrationPoint(
                    i, reference_point_rt, predicted_reference_point_rt,
                    np.array(relative_retention_time), np.array(predicted_relative_retention_time),
                    np.array(resid), np.log10(reference_point.total_signal)
                )

    def _predict(self):
        configs = self.configurations
        predicted_apex_time_array = []
        weight = []
        for key, calibration_point in configs.items():
            predicted_apex_time_array.append(
                (calibration_point.prrt + calibration_point.reference_point_rt) * calibration_point.weight)
            weight.append(calibration_point.weight)
        return np.sum(predicted_apex_time_array, axis=0) / np.sum(weight)

    def _df(self):
        return max(len(self.configurations) - len(self.model.parameters), 1)

    def _score(self):
        score = stats.t.sf(
            abs(self.predicted_apex_time_array - self.apex_time_array),
            df=self._df(), scale=self.scale) * 2
        score -= 1e-3
        score[score < 1e-3] = 1e-3
        return score

    def R2(self, adjust=False):
        y = self.apex_time_array
        w = self.weight_array
        yhat = self.predicted_apex_time_array
        residuals = (y - yhat)
        rss = (w * residuals * residuals).sum()
        tss = (y - y.mean())
        tss = (w * tss * tss).sum()
        if adjust:
            n = len(y)
            k = len(self.model.parameters)
            adjustment_factor = (n - 1.0) / float(n - k - 1.0)
        else:
            adjustment_factor = 1.0
        R2 = (1 - adjustment_factor * (rss / tss))
        return R2

    @classmethod
    def predict(cls, training_examples, testing_examples, model, dilation=1.0):
        inst = cls(training_examples, testing_examples, model, dilation)
        return inst.predicted_apex_time_array

    @classmethod
    def score(cls, training_examples, testing_examples, model, dilation=1.0):
        inst = cls(training_examples, testing_examples, model, dilation)
        return inst.score_array

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = self.apex_time_array
        Y = self.predicted_apex_time_array
        S = np.array([c.total_signal for c in self.testing_examples])
        S /= S.max()
        S *= 100
        ax.scatter(X, Y, s=S, marker='o')
        ax.plot((X.min(), X.max()), (X.min(), X.max()), color='black', linestyle='--', lw=0.75)
        ax.set_xlabel('Experimental Apex Time', fontsize=18)
        ax.set_ylabel('Predicted Apex Time', fontsize=18)
        ax.figure.text(0.8, 0.15, "$R^2=%0.4f$" % self.R2(), ha='center')
        return ax

    def filter(self, threshold):
        filtered = self.__class__(
            self.training_examples,
            self.testing_examples[self.score_array > threshold],
            self.model,
            self.scale,
            self.dilation,
            self.weighted)
        return filtered
