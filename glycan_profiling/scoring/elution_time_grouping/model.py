import csv
import itertools
import logging

from numbers import Number
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np
from scipy import stats

from glycopeptidepy import PeptideSequence, parse

from glypy.utils import make_counter
from glypy.structure.glycan_composition import HashableGlycanComposition

from glycan_profiling import chromatogram_tree
from glycan_profiling.database.composition_network.space import composition_distance

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycan_profiling.task import TaskBase

from glycan_profiling.chromatogram_tree import Unmodified, Ammonium
from glycan_profiling.scoring.base import ScoringFeatureBase

from .structure import _get_apex_time, GlycopeptideChromatogramProxy, GlycoformAggregator, DeltaOverTimeFilter
from .linear_regression import (ransac, weighted_linear_regression_fit, prediction_interval, SMALL_ERROR, weighted_linear_regression_fit_ridge)
from .reviser import (IntervalModelReviser, IsotopeRule, AmmoniumMaskedRule,
                      AmmoniumUnmaskedRule, HexNAc2Fuc1NeuAc2ToHex6AmmoniumRule, IsotopeRule2)

logger = logging.getLogger("glycan_profiling.elution_time_model")
logger.addHandler(logging.NullHandler())


class IntervalRange(object):
    def __init__(self, lower=None, upper=None):
        if isinstance(lower, IntervalRange):
            self.lower = lower.lower
            self.upper = lower.upper
        elif isinstance(lower, (tuple, list)):
            self.lower, self.upper = lower
        else:
            self.lower = lower
            self.upper = upper

    def clamp(self, value):
        if self.lower is None:
            return value
        if value < self.lower:
            return self.lower
        if value > self.upper:
            return self.upper
        return value

    def interval(self, value):
        center = np.mean(value)
        lower = center - self.clamp(abs(center - value[0]))
        upper = center + self.clamp(abs(value[1] - center))
        return [lower, upper]

    def __repr__(self):
        return "{self.__class__.__name__}({self.lower}, {self.upper})".format(self=self)


class AbundanceWeightedMixin(object):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            1.0 / (x.total_signal * x.weight) for x in self.chromatograms
        ]
        if len(self.chromatograms) == 0:
            return W
        W /= W.max()
        return W


class ChromatgramFeatureizerBase(object):

    transform = None

    def feature_names(self):
        return ['intercept', 'mass']

    def _get_apex_time(self, chromatogram):
        t = _get_apex_time(chromatogram)
        if self.transform is None:
            return t
        return t - self.transform(chromatogram)

    def _prepare_data_vector(self, chromatogram):
        return np.array([1, chromatogram.weighted_neutral_mass, ])

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
        )).T

    def build_weight_matrix(self):
        return 1.0 / np.diag([x.weight for x in self.chromatograms])


class PredictorBase(object):
    transform = None

    def predict(self, chromatogram):
        t = self._predict(self._prepare_data_vector(chromatogram))
        if self.transform is None:
            return t
        return t + self.transform(chromatogram)

    def _predict(self, x):
        return x.dot(self.parameters)

    def predict_interval(self, chromatogram, alpha=0.05):
        x = self._prepare_data_vector(chromatogram)
        return self._predict_interval(x, alpha=alpha)

    def _predict_interval(self, x, alpha=0.05):
        y = self._predict(x)
        return prediction_interval(self.solution, x, y, alpha=alpha)

    def __call__(self, x):
        return self.predict(x)


class LinearModelBase(PredictorBase, SpanningMixin):
    def _init_model_data(self):
        self.neutral_mass_array = np.array([
            x.weighted_neutral_mass for x in self.chromatograms
        ])
        self.data = self._prepare_data_matrix(self.neutral_mass_array)

        self.apex_time_array = np.array([
            self._get_apex_time(x) for x in self.chromatograms
        ])

        self.weight_matrix = self.build_weight_matrix()

        self.parameters = None
        self.residuals = None
        self.estimate = None
        self._update_model_time_range()

    def _update_model_time_range(self):
        self.start = self.apex_time_array.min()
        self.end = self.apex_time_array.max()
        d = np.diag(self.weight_matrix)
        self.centroid = self.apex_time_array.dot(d) / d.sum()

    @property
    def start_time(self):
        return self.start

    @property
    def end_time(self):
        return self.end

    def _fit(self, resample=False, alpha=None):
        if resample:
            solution = ransac(self.data, self.apex_time_array,
                              self.weight_matrix, regularize_alpha=alpha)
            if alpha is None:
                alt = weighted_linear_regression_fit(
                    self.data, self.apex_time_array, self.weight_matrix)
            else:
                alt = weighted_linear_regression_fit_ridge(
                    self.data, self.apex_time_array, self.weight_matrix, alpha)
            if alt.R2 > solution.R2:
                return alt
            return solution
        else:
            if alpha is None:
                solution = weighted_linear_regression_fit(
                    self.data, self.apex_time_array, self.weight_matrix)
            else:
                solution = weighted_linear_regression_fit_ridge(
                    self.data, self.apex_time_array, self.weight_matrix, alpha)
        return solution

    def default_regularization(self):
        p = self.data.shape[1]
        return np.ones_like(p) * 0.001

    def fit(self, resample=False, alpha=None):
        solution = self._fit(resample=resample, alpha=alpha)
        self.estimate = solution.yhat
        self.residuals = solution.residuals
        self.parameters = solution.parameters
        self.projection_matrix = solution.projection_matrix
        self.solution = solution
        return self

    def loglikelihood(self):
        n = self.data.shape[0]
        n2 = n / 2.0
        rss = self.solution.rss
        # The "concentrated likelihood"
        likelihood = -np.log(rss) * n2
        # The likelihood constant
        likelihood -= (1 + np.log(np.pi / n2)) / n2
        likelihood += 0.5 * np.sum(np.log(np.diag(self.weight_matrix)))
        return likelihood

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
        XtWX_inv = np.linalg.pinv(
            (self.data.T.dot(self.weight_matrix).dot(self.data)))
        # With unknown variance, use the mean squared error estimate
        sigma_params = np.sqrt(np.diag(self.mse * XtWX_inv))
        degrees_of_freedom = len(self.apex_time_array) - \
            len(self.parameters) - 1.0
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

    def R2(self, adjust=True):
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
            adjustment_factor = (n - 1.0) / max(float(n - k - 1.0), 1)
        else:
            adjustment_factor = 1.0
        R2 = (1 - adjustment_factor * (rss / tss))
        return R2

    def _df(self):
        return max(len(self.chromatograms) - len(self.parameters), 1)


class ElutionTimeFitter(LinearModelBase, ChromatgramFeatureizerBase, ScoringFeatureBase):
    feature_type = 'elution_time'

    def __init__(self, chromatograms, scale=1, transform=None, width_range=None, regularize=False):
        self.chromatograms = chromatograms
        self.neutral_mass_array = None
        self.data = None
        self.apex_time_array = None
        self.weight_matrix = None
        self.parameters = None
        self.residuals = None
        self.estimate = None
        self.scale = scale
        self.transform = transform
        self.width_range = IntervalRange(width_range)
        self.regularize = regularize
        self._init_model_data()

    def _get_chromatograms(self):
        return self.chromatograms

    def score(self, chromatogram):
        apex = self.predict(chromatogram)
        # Use heavier tails (scale 2) to be more tolerant of larger chromatographic
        # errors.
        # The survival function's maximum value is 0.5, so double this to map the
        # range of values to be (0, 1)
        score = stats.t.sf(
            abs(apex - self._get_apex_time(chromatogram)),
            df=self._df(), scale=self.scale) * 2
        return max((score - SMALL_ERROR), SMALL_ERROR)

    def score_interval(self, chromatogram, alpha=0.05):
        interval = self.predict_interval(chromatogram, alpha=alpha)
        pred = interval.mean()
        delta = abs(chromatogram.apex_time - pred)
        width = (interval[1] - interval[0]) / 2.0
        if self.width_range is not None:
            if np.isnan(width):
                width = self.width_range.upper
            else:
                width = self.width_range.clamp(width)
        return max(1 - delta / width, 0.0)

    def calibrate_prediction_interval(self, chromatograms=None, alpha=0.05):
        if chromatograms is None:
            chromatograms = self.chromatograms
        ivs = np.array([self.predict_interval(c, alpha) for c in chromatograms])
        widths = (ivs[:, 1] - ivs[:, 0]) / 2.0
        self.width_range = IntervalRange(*np.quantile(widths, [0.25, 0.75]))
        return self

    def plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        ax.scatter(self.neutral_mass_array,
                   self.apex_time_array, label='Observed')
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

    def summary(self, join_char=' | ', justify=True):
        if justify:
            formatter = str.ljust
        else:
            def formatter(x, y):
                return x
        column_labels = ['Feature Name', "Value", "p-value", "Conf. Int."]
        feature_names = list(map(str, self.feature_names()))
        parameter_values = ['%0.2f' % val for val in self.parameters]
        signif = ['%0.3f' % val for val in self.parameter_significance()]
        ci = ['%0.2f-%0.2f' % tuple(cinv)
              for cinv in self.parameter_confidence_interval()]
        sizes = list(map(len, column_labels))
        value_sizes = [max(map(len, col))
                       for col in [feature_names, parameter_values, signif, ci]]
        sizes = map(max, zip(sizes, value_sizes))
        table = [[formatter(v, sizes[i]) for i, v in enumerate(column_labels)]]
        for row in zip(feature_names, parameter_values, signif, ci):
            table.append([
                formatter(v, sizes[i]) for i, v in enumerate(row)
            ])
        joiner = join_char.join
        table_str = '\n'.join(map(joiner, table))
        return table_str

    def to_csv(self, fh):
        writer = csv.writer(fh)
        writer.writerow(["name", "value"])
        for fname, value in zip(self.feature_names(), self.parameters):
            writer.writerow([fname, value])

    def to_dict(self):
        out = OrderedDict()
        keys = self.feature_names()
        for k, v in zip(keys, self.parameters):
            out[k] = v
        return out

    def _infer_factors(self):
        keys = set()
        for record in self.chromatograms:
            keys.update(record.glycan_composition)
        keys = sorted(map(str, keys))
        return keys

    def plot_residuals(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        ax.scatter(self.apex_time_array, self.residuals, s=np.diag(self.weight_matrix) * 1000.0, alpha=0.25,
                   edgecolor='black')
        return ax


class SimpleLinearFitter(LinearModelBase):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = self.apex_time_array = np.array(y)
        self.weight_matrix = np.diag(np.ones_like(y))
        self.data = np.stack((np.ones_like(x), x)).T

    def predict(self, vx):
        if not isinstance(vx, (list, tuple, np.ndarray)):
            vx = [vx]
        return np.stack((np.ones_like(vx), vx)).T.dot(self.parameters)

    def _prepare_data_vector(self, x):
        return np.array([1., x])


class AbundanceWeightedElutionTimeFitter(AbundanceWeightedMixin, ElutionTimeFitter):
    pass


class FactorChromatogramFeatureizer(ChromatgramFeatureizerBase):
    def feature_names(self):
        return ['intercept'] + self.factors

    def _prepare_data_matrix(self, mass_array):
        return np.vstack([np.ones(len(mass_array)), ] + [
            np.array([c.glycan_composition[f] for c in self.chromatograms])
            for f in self.factors]).T

    def _prepare_data_vector(self, chromatogram, no_intercept=False):
        intercept = 0 if no_intercept else 1
        return np.array(
            [intercept] + [
                chromatogram.glycan_composition[f] for f in self.factors])


class FactorTransform(PredictorBase, FactorChromatogramFeatureizer):
    def __init__(self, factors, parameters, intercept=0.0):
        self.factors = factors
        # Add a zero intercept
        self.parameters = np.concatenate([[intercept], parameters])


class FactorElutionTimeFitter(FactorChromatogramFeatureizer, ElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None, regularize=False):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self.factors = list(factors)
        super(FactorElutionTimeFitter, self).__init__(
            chromatograms, scale=scale, transform=transform,
            width_range=width_range, regularize=regularize)

    def predict_delta_glycan(self, chromatogram, delta_glycan):
        try:
            shifted = chromatogram.shift_glycan_composition(delta_glycan)
        except AttributeError:
            shifted = GlycopeptideChromatogramProxy.from_obj(
                chromatogram).shift_glycan_composition(delta_glycan)
        return self.predict(shifted)

    def prediction_plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        ax.scatter(self.apex_time_array, self.estimate)
        preds = np.array(self.estimate)
        obs = np.array(self.apex_time_array)
        lo, hi = min(preds.min(), obs.min()), max(preds.max(), obs.max())
        lo -= 0.2
        hi += 0.2
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', lw=0.75)
        ax.set_xlabel("Experimental RT (Min)", fontsize=14)
        ax.set_ylabel("Predicted RT (Min)", fontsize=14)
        ax.figure.text(0.15, 0.8, "$R^2:%0.2f$\nMSE:%0.2f" % (self.R2(True), self.mse))
        return ax

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
            partitions[key].append(
                (self.neutral_mass_array[i], self.apex_time_array[i]))

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

    def predict(self, chromatogram, no_intercept=False):
        return self._predict(self._prepare_data_vector(chromatogram, no_intercept=no_intercept))


class AbundanceWeightedFactorElutionTimeFitter(AbundanceWeightedMixin, FactorElutionTimeFitter):
    pass


class PeptideBackboneKeyedMixin(object):
    def get_peptide_key(self, chromatogram):
        return str(PeptideSequence(str(chromatogram.structure)).deglycosylate())


class PeptideGroupChromatogramFeatureizer(FactorChromatogramFeatureizer, PeptideBackboneKeyedMixin):
    def _prepare_data_matrix(self, mass_array):
        p = len(self._peptide_to_indicator)
        n = len(self.chromatograms)
        peptides = np.zeros((p, n))
        indicator = dict(self._peptide_to_indicator)
        for i, c in enumerate(self.chromatograms):
            try:
                j = indicator[self.get_peptide_key(c)]
                peptides[j, i] = 1
            except KeyError:
                pass
        # Omit the intercept, so that all peptide levels are used without inducing linear dependence.
        return np.vstack([peptides, ] +
                         [np.array([c.glycan_composition[f] for c in self.chromatograms]) for f in self.factors]).T

    def feature_names(self):
        names = []
        peptides = [None] * len(self._peptide_to_indicator)
        for key, value in self._peptide_to_indicator.items():
            peptides[value] = key
        names.extend(peptides)
        names.extend(self.factors)
        return names

    def _prepare_data_vector(self, chromatogram, no_intercept=False):
        p = len(self._peptide_to_indicator)
        peptides = [0 for _ in range(p)]
        indicator = dict(self._peptide_to_indicator)
        if not no_intercept:
            try:
                peptide_key = self.get_peptide_key(chromatogram)
                peptides[indicator[peptide_key]] = 1
            except KeyError:
                logger.debug(
                    "Peptide sequence of %s not part of the model.", chromatogram)
        return np.array(
            peptides + [chromatogram.glycan_composition[f] for f in self.factors])

    def has_peptide(self, peptide):
        '''Check if the peptide is included in the model.

        Parameters
        ----------
        peptide : str
            The peptide sequence to check

        Returns
        -------
        bool
        '''
        if not isinstance(peptide, str):
            peptide = self.get_peptide_key(peptide)
        return peptide in self._peptide_to_indicator


class PeptideFactorElutionTimeFitter(PeptideGroupChromatogramFeatureizer, FactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None, regularize=False):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self._peptide_to_indicator = defaultdict(make_counter(0))
        self.by_peptide = defaultdict(list)
        self.peptide_groups = []
        # Ensure that _peptide_to_indicator is properly initialized
        for obs in chromatograms:
            key = self.get_peptide_key(obs)
            self.peptide_groups.append(self._peptide_to_indicator[key])
            self.by_peptide[key].append(obs)
        self.peptide_groups = np.array(self.peptide_groups)
        super(PeptideFactorElutionTimeFitter, self).__init__(
            chromatograms, list(factors), scale=scale, transform=transform,
            width_range=width_range, regularize=regularize)

    @property
    def peptide_count(self):
        return len(self.by_peptide)

    def default_regularization(self):
        monosaccharide_alphas = {}
        alpha = np.concatenate((
            np.zeros(self.peptide_count),
            [monosaccharide_alphas.get(f, 0.01) for f in self.factors]
        ))
        return alpha

    def fit(self, resample=False, alpha=None):
        if alpha is None and self.regularize:
            alpha = self.default_regularization()
        return super(PeptideGroupChromatogramFeatureizer, self).fit(resample, alpha)

    def groupwise_R2(self, adjust=True):
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss_u = (np.diag(w) * residuals * residuals)
        tss = (y - y.mean())
        tss_u = (np.diag(w) * tss * tss)

        mapping = {}
        for key, value in self._peptide_to_indicator.items():
            mask = x[:, value] == 1
            rss = rss_u[mask].sum()
            tss = tss_u[mask].sum()
            n = len(y)
            k = len(self.parameters)
            if adjust:
                adjustment_factor = (n - 1.0) / float(n - k - 1.0)
            else:
                adjustment_factor = 1.0
            R2 = (1 - adjustment_factor * (rss / tss))
            mapping[key] = R2
        return mapping

    def plot_residuals(self, ax=None, subset=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        if subset is not None and subset:
            if isinstance(subset[0], int):
                group_ids = subset
            else:
                group_ids = [self._peptide_to_indicator[k] for k in subset]
        else:
            group_ids = np.unique(self.peptide_groups)
        group_label_map = {v: k for k, v in self._peptide_to_indicator.items()}
        preds = np.array(self.estimate)
        obs = np.array(self.apex_time_array)
        weights = np.diag(self.weight_matrix)
        for i in group_ids:
            mask = self.peptide_groups == i
            dv = preds[mask] - obs[mask]
            sub = obs[mask]
            a = ax.scatter(
                sub, dv,
                s=weights[mask] * 500,
                edgecolors='black',
                alpha=0.5,
                label="%s %0.2f" % (group_label_map[i], sub.min()))
            f = SimpleLinearFitter(obs[mask], dv).fit()
            x = np.linspace(sub.min(), sub.max())
            ax.plot(x, f.predict(x), color=a.get_facecolor()[0])

        return ax

    def prediction_plot(self, ax=None, subset=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        if subset is not None and subset:
            if isinstance(subset[0], int):
                group_ids = subset
            else:
                group_ids = [self._peptide_to_indicator[k] for k in subset]
        else:
            group_ids = np.unique(self.peptide_groups)
        group_label_map = {v: k for k, v in self._peptide_to_indicator.items()}
        preds = np.array(self.estimate)
        obs = np.array(self.apex_time_array)
        weights = np.diag(self.weight_matrix)
        for i in group_ids:
            mask = self.peptide_groups == i
            ax.scatter(
                obs[mask], preds[mask],
                s=weights[mask] * 500,
                edgecolors='black',
                alpha=0.5,
                label="%s %0.2f" % (group_label_map[i], obs[mask].min()))

        lo, hi = min(preds.min(), obs.min()), max(preds.max(), obs.max())
        lo -= 0.2
        hi += 0.2
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', lw=0.75)
        ax.set_xlabel("Experimental RT (Min)", fontsize=14)
        ax.set_ylabel("Predicted RT (Min)", fontsize=14)
        ax.figure.text(0.15, 0.8, "$R^2:%0.2f$\nMSE:%0.2f" %
                       (self.R2(True), self.mse))
        return ax


class AbundanceWeightedPeptideFactorElutionTimeFitter(AbundanceWeightedMixin, PeptideFactorElutionTimeFitter):
    pass


class ModelEnsemble(PeptideBackboneKeyedMixin):
    def __init__(self, models, width_range=None):
        self.models = models
        self._models = list(models.values())
        self.width_range = IntervalRange(width_range)
        self._chromatograms = None
        self.external_peptide_offsets = {}

    def _models_for(self, chromatogram):
        if not isinstance(chromatogram, Number):
            point = chromatogram.apex_time
        else:
            point = chromatogram
        for model in self._models:
            if model.contains(point):
                weight = abs(model.centroid - point) + 1
                yield model, 1.0 / weight

    def coverage_for(self, chromatogram):
        refs = []
        weights = []
        for mod, weight in self._models_for(chromatogram):
            refs.append(mod.has_peptide(chromatogram))
            weights.append(weight)
        coverage = np.dot(refs, weights) / np.sum(weights)
        return coverage

    def predict_interval_external_peptide(self, chromatogram, alpha=0.05, merge=True):
        key = self.get_peptide_key(chromatogram)
        offset = self.external_peptide_offsets[key]
        interval = self.predict_interval(chromatogram, alpha=alpha, merge=merge)
        if merge:
            interval += offset
        else:
            interval = [iv + offset for iv in interval]
        return interval

    def predict_interval(self, chromatogram, alpha=0.05, merge=True, check_peptide=True):
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            if check_peptide and not mod.has_peptide(chromatogram):
                continue
            preds.append(mod.predict_interval(chromatogram, alpha=alpha))
            weights.append(w)
        weights = np.array(weights)
        if len(weights) == 0:
            return np.array([np.nan, np.nan])

        mask = ~np.isnan(weights) & ~(np.isnan(preds).sum(axis=1).astype(bool))
        preds = np.array(preds)
        preds = preds[mask, :]
        weights = weights[mask]
        weights /= weights.max()
        if merge:
            if len(weights) == 0:
                return np.array([np.nan, np.nan])
            avg = np.average(preds, weights=weights, axis=0)
            return avg
        return preds, weights

    def score_interval(self, chromatogram, alpha=0.05):
        interval = self.predict_interval(chromatogram, alpha=alpha)
        pred = interval.mean()
        delta = abs(chromatogram.apex_time - pred)
        width = (interval[1] - interval[0]) / 2.0
        if self.width_range is not None:
            if np.isnan(width):
                width = self.width_range.upper
            else:
                width = self.width_range.clamp(width)
        return max(1 - delta / width, 0.0)

    def predict(self, chromatogram, merge=True, check_peptide=True):
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            if check_peptide and not mod.has_peptide(chromatogram):
                continue
            preds.append(mod.predict(chromatogram))
            weights.append(w)
        if len(weights) == 0:
            return 0
        weights = np.array(weights)
        weights /= weights.max()
        if merge:
            avg = np.average(preds, weights=weights)
            return avg
        return np.array(preds), weights

    def score(self, chromatogram, merge=True):
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            preds.append(mod.score(chromatogram))
            weights.append(w)
        weights = np.array(weights)
        weights /= weights.max()
        if merge:
            avg = np.average(preds, weights=weights)
            return avg
        preds = np.array(preds)
        return preds, weights

    def _get_chromatograms(self):
        if self._chromatograms:
            return self._chromatograms
        chroma = set()
        for model in self.models.values():
            for chrom in model._get_chromatograms():
                chroma.add(chrom)
        self._chromatograms = sorted(chroma, key=lambda x: x.apex_time)
        return self._chromatograms

    @property
    def chromatograms(self):
        return self._get_chromatograms()

    def calibrate_prediction_interval(self, chromatograms=None, alpha=0.05):
        if chromatograms is None:
            chromatograms = self.chromatograms
        ivs = np.array([self.predict_interval(c, alpha)
                        for c in chromatograms])
        widths = (ivs[:, 1] - ivs[:, 0]) / 2.0
        widths = widths[~np.isnan(widths)]
        self.width_range = IntervalRange(*np.quantile(widths, [0.25, 0.75]))
        if np.isnan(self.width_range.lower):
            raise ValueError("Width range cannot be NaN")
        return self


def unmodified_modified_predicate(x):
    return Unmodified in x.mass_shifts and Ammonium in x.mass_shifts


class EnsembleBuilder(TaskBase):
    def __init__(self, aggregator):
        self.aggregator = aggregator
        self.points = None
        self.centers = OrderedDict()
        self.models = OrderedDict()

    def estimate_width(self):
        try:
            width = int(max(np.nanmedian(v) for v in
                            self.aggregator.deltas().values()
                            if len(v) > 0)) + 1
        except ValueError:
            width = 2.0
        width *= 2
        return float(width)

    def generate_points(self, width):
        points = np.arange(
            int(self.aggregator.start_time) - 1,
            int(self.aggregator.end_time) + 1, width / 4.)
        if len(points) > 0 and points[-1] <= self.aggregator.end_time:
            points = np.append(points, self.aggregator.end_time + 1)
        return points

    def partition(self, width=None, predicate=None):
        if width is None:
            width_value = self.estimate_width()
            width = defaultdict(lambda: width_value)
        elif isinstance(width, (float, int, np.ndarray)):
            width_value = width
            width = defaultdict(lambda: width_value)
        else:
            width_value = None
        if predicate is None:
            predicate = bool
        if self.points is None:
            assert width_value is not None
            self.points = self.generate_points(width_value)
        centers = OrderedDict()
        for point in self.points:
            w = width[point]
            centers[point] = GlycoformAggregator(
                list(filter(predicate,
                            self.aggregator.overlaps(point - w, point + w))))
        self.centers = centers
        return self

    def estimate_delta_widths(self, baseline=None):
        if baseline is None:
            baseline = 0.0

        delta_index = defaultdict(list)
        time = []
        for cent, group in self.centers.items():
            time.append(cent)
            deltas = group.deltas()

            for factor in self.aggregator.factors:
                val = np.nanmedian(deltas.get(factor, []))
                delta_index[factor].append(val)

        delta_index = {
            k: DeltaOverTimeFilter(k, v, time) for k, v in delta_index.items()
        }

        group_widths = OrderedDict()
        for t in time:
            best = 0
            for k, v in delta_index.items():
                val = v[v.search(t)]
                best = max(abs(val), best)
            best = round(best) * 2
            if best < 1:
                best = 2
            if best < baseline:
                best = baseline
            group_widths[t] = best
        return group_widths

    def fit(self, predicate=None, model_type=None, regularize=False, resample=False):
        if model_type is None:
            model_type = AbundanceWeightedPeptideFactorElutionTimeFitter
        if predicate is None:
            predicate = bool
        models = OrderedDict()
        point_members = list(self.centers.items())
        n = len(point_members)
        for i, (point, members) in enumerate(point_members):
            obs = list(filter(predicate, members))
            if not obs:
                continue
            m = models[point] = model_type(
                obs, self.aggregator.factors, regularize=regularize)
            if m.data.shape[0] <= m.data.shape[1]:
                offset = 1
                while m.data.shape[0] <= m.data.shape[1] and offset < (len(point_members) // 2 + 1):
                    extra = set(members)
                    for j in range(1, offset + 1):
                        if i > j:
                            extra.update(list(point_members[i - j][1]))
                        if i + j < n:
                            extra.update(list(point_members[i + j][1]))
                    obs = sorted(filter(predicate, extra), key=lambda x: x.apex_time)
                    m = models[point] = model_type(obs, self.aggregator.factors, regularize=regularize)
                    offset += 1
                m = models[point]

            m.fit(resample=resample)
        self.models = models
        return self

    def merge(self):
        return ModelEnsemble(self.models)


DeltaParam = namedtuple(
    "DeltaParam", ("mean", "sdev", "size", "values", "weights"))


class LocalShiftGraph(object):
    def __init__(self, chromatograms, max_distance=3):
        self.chromatograms = chromatograms
        self.max_distance = max_distance
        self.edges = defaultdict(list)
        self.shift_to_delta = defaultdict(list)
        self.groups = GlycoformAggregator(chromatograms)

        self.build_edges()
        self.delta_params = self.estimate_delta_distribution()

    def build_edges(self):
        for key, group in self.groups.by_peptide.items():
            base_time = 0.0
            for a, b in itertools.permutations(group, 2):
                delta_comp = a.glycan_composition - b.glycan_composition
                distance = composition_distance(
                    a.glycan_composition, b.glycan_composition)[0]
                if distance > self.max_distance or distance == 0:
                    continue
                structure = parse(key + str(delta_comp))
                rec = GlycopeptideChromatogramProxy(
                    a.weighted_neutral_mass, base_time + a.apex_time - b.apex_time,
                    np.sqrt(a.total_signal * b.total_signal),
                    delta_comp, structure=structure,
                    weight=float(distance) ** 4 + (a.weight + b.weight)
                )
                edge_key = frozenset((a.structure, b.structure))
                delta_key = delta_comp
                # Probably don't need this
                self.edges[edge_key].append(rec)
                # Use string form to discard 0-value keys
                self.shift_to_delta[str(delta_key)].append((rec, edge_key))

    def estimate_delta_distribution(self):
        params = {}
        for shift, delta_chroms in self.shift_to_delta.items():
            apex_times = []
            weights = []
            for rec, _key in delta_chroms:
                apex_times.append(rec.apex_time)
                weights.append(rec.total_signal)

            apex_times = np.array(apex_times)
            weights = np.log10(weights)

            weighted_mean = np.dot(apex_times, weights) / weights.sum()
            weighted_sdev = np.sqrt(weights.dot((apex_times - weighted_mean) ** 2) / (weights.sum() - 1))
            if weighted_sdev > abs(weighted_mean):
                tmp = weighted_sdev
                weighted_sdev = max(abs(weighted_mean), 1.0)
                logger.debug("Delta parameter {}'s standard deviation was over-dispersed {}/{}. Capping at {}".format(
                    shift, tmp, weighted_mean, weighted_sdev))

            # NOTE: When len(delta_chroms) == 1, weighted_sdev = 0.0, so no observations
            # are allowed through. This is intentional because we cannot trust singletons here.
            upper = weighted_mean + weighted_sdev * 2.5
            lower = weighted_mean - weighted_sdev * 2.5
            ii = []
            for i, x in enumerate(apex_times):
                if lower < x < upper:
                    ii.append(i)
            if len(ii) != len(apex_times):
                apex_times = apex_times[ii]
                weights = weights[ii]
                weighted_mean = np.dot(apex_times, weights) / weights.sum()
                weighted_sdev = np.sqrt(weights.dot((apex_times - weighted_mean) ** 2) / (weights.sum() - 1))

            params[shift] = DeltaParam(weighted_mean, weighted_sdev, len(apex_times), apex_times, weights)
        return params

    def is_composite(self, shift):
        value = HashableGlycanComposition.parse(shift)
        return sum(map(abs, value.values())) > 1

    def composite_to_individuals(self, shift):
        shift = HashableGlycanComposition.parse(shift)
        return [(HashableGlycanComposition({k: v / abs(v)}), v) for k, v in shift.items()]

    def compose_shift(self, shift):
        parts = self.composite_to_individuals(shift)
        mean = 0.0
        variance = 0.0
        for (part, mult) in parts:
            # Throw a KeyError if we do not have a component edge
            # to reference.
            param = self.delta_params[part]
            mean += param.mean * abs(mult)
            variance += param.sdev ** 2 * abs(mult)
        return mean, np.sqrt(variance)

    def process_edges(self):
        bad_edges = set()
        for shift, delta_chroms in self.shift_to_delta.items():
            params = self.delta_params[shift]

            spread = params.sdev * 2
            lo = params.mean - spread
            hi = params.mean + spread


            bad = False
            if self.is_composite(shift):
                try:
                    comp_mean, comp_sdev = self.compose_shift(shift)
                    maybe_bad = not (comp_mean - (comp_sdev * 1.2) < params.mean < comp_mean + (comp_sdev * 1.2))
                    if maybe_bad and params.size < 3:
                        bad = True
                except KeyError as err:
                    # We don't have a component edge to support this composite edge, so we don't
                    # have enough information in the training data, so we're unable QC these edges.
                    # This is different from when we have a singular observation of any component,
                    # which may be a logical failure here.
                    logger.debug("Missing component key %r", err)

            if bad:
                for delta_chrom, edge_key in delta_chroms:
                    bad_edges.add(edge_key)
            else:
                for delta_chrom, edge_key in delta_chroms:
                    if delta_chrom.apex_time < lo or delta_chrom.apex_time > hi:
                        bad_edges.add(edge_key)
        pruned = []
        for e in bad_edges:
            pruned.extend(self.edges.pop(e))
        return pruned

    def select_edges(self):
        edges = []
        for es in self.edges.values():
            edges.extend(es)
        return edges


class RelativeShiftFactorElutionTimeFitter(AbundanceWeightedPeptideFactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None, regularize=False, max_distance=3):
        recs, groups = self.build_deltas(chromatograms, max_distance=max_distance)
        self.groups = groups
        self.peptide_offsets = {}
        self.max_distance = max_distance
        super(RelativeShiftFactorElutionTimeFitter, self).__init__(
            recs, factors, scale=scale, transform=transform,
            width_range=width_range, regularize=regularize)

    def build_deltas(self, chromatograms, max_distance):
        recs = []
        groups = GlycoformAggregator(chromatograms)
        for key, group in groups.by_peptide.items():
            base_time = 0.0
            for a, b in itertools.permutations(group, 2):
                delta_comp = a.glycan_composition - b.glycan_composition
                distance = composition_distance(
                    a.glycan_composition, b.glycan_composition)[0]
                if distance > max_distance or distance == 0:
                    continue
                structure = parse(key + str(delta_comp))
                rec = GlycopeptideChromatogramProxy(
                    a.weighted_neutral_mass, base_time + a.apex_time - b.apex_time,
                    np.sqrt(a.total_signal * b.total_signal),
                    delta_comp, structure=structure,
                    weight=float(distance) ** 4 + (a.weight + b.weight)
                )
                recs.append(rec)
        assert len(recs) > 0
        return recs, groups

    def _get_chromatograms(self):
        return list(self.groups)

    def fit(self, *args, **kwargs):
        super(RelativeShiftFactorElutionTimeFitter, self).fit(*args, **kwargs)
        for key, group in self.groups.by_peptide.items():
            remainders = []
            weights = []
            for case in group:
                offset = self.predict(case)
                remainders.append(case.apex_time - offset)
                weights.append(case.total_signal)
            remainders = np.array(remainders)
            self.peptide_offsets[key] = np.average(remainders, weights=weights)
        return self

    def predict(self, chromatogram):
        time = super(RelativeShiftFactorElutionTimeFitter,
                     self).predict(chromatogram)
        key = self.get_peptide_key(chromatogram)
        time += self.peptide_offsets.get(key, 0)
        return time

    def predict_interval(self, chromatogram, *args, **kwargs):
        iv = super(RelativeShiftFactorElutionTimeFitter,
                   self).predict_interval(chromatogram)
        key = self.get_peptide_key(chromatogram)
        iv += self.peptide_offsets.get(key, 0)
        return iv

    def calibrate_prediction_interval(self, chromatograms=None, alpha=0.05):
        return super(RelativeShiftFactorElutionTimeFitter, self).calibrate_prediction_interval(list(self.groups), alpha)

    def _update_model_time_range(self):
        apexes = []
        weights = []
        for case in self.groups:
            t = case.apex_time
            w = case.total_signal
            apexes.append(t)
            weights.append(w)
        apexes = np.array(apexes)
        self.start = apexes.min()
        self.end = apexes.max()
        self.centroid = apexes.dot(weights) / np.sum(weights)

    # This omission is by design, it prevents peptide backbones that did
    # not have a pair from being considered.
    #
    # def has_peptide(self, peptide):
    #     if not isinstance(peptide, str):
    #         peptide = self.get_peptide_key(peptide)
    #     return peptide in self.peptide_offsets


class LocalOutlierFilteringRelativeShiftFactorElutionTimeFitter(RelativeShiftFactorElutionTimeFitter):
    def build_deltas(self, chromatograms, max_distance):
        graph = LocalShiftGraph(chromatograms, max_distance)
        graph.process_edges()
        groups = graph.groups
        recs = graph.select_edges()
        return recs, groups


def mask_in(full_set, incoming):
    incoming_map = {i.tag: i for i in incoming}
    out = []
    for chrom in full_set:
        if chrom.tag in incoming_map:
            out.append(incoming_map[chrom.tag])
        else:
            out.append(chrom)
    return out


def model_type_dispatch(self, chromatogams, factors=None, scale=1, transform=None, width_range=None, regularize=False):
    try:
        return RelativeShiftFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize, max_distance=3)
    except AssertionError:
        return AbundanceWeightedPeptideFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize)


def model_type_dispatch_outlier_remove(self, chromatogams, factors=None, scale=1, transform=None, width_range=None, regularize=False):
    try:
        return LocalOutlierFilteringRelativeShiftFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize, max_distance=3)
    except AssertionError:
        return AbundanceWeightedPeptideFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize)


class ModelBuildingPipeline(TaskBase):
    '''A local regression-based ensemble of overlapping
    retention time models.
    '''

    model_type = model_type_dispatch_outlier_remove

    def __init__(self, aggregate, calibration_alpha=0.001, valid_glycans=None):
        self.aggregate = aggregate
        self.calibration_alpha = calibration_alpha
        self._current_model = None
        self.model_history = []
        self.revision_history = []
        self.revised_tags = set()
        self.valid_glycans = valid_glycans

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, value):
        if self._current_model is not None and value is not self._current_model:
            self.model_history.append(self._current_model)
        self._current_model = value

    def fit_first_model(self, regularize=False):
        model = self.fit_model(
            self.aggregate,
            unmodified_modified_predicate,
            regularize=regularize)
        return model

    def fit_model(self, aggregate, predicate, regularize=False, resample=False):
        builder = EnsembleBuilder(aggregate)
        w = builder.estimate_width()
        # If a delta width is allowed, approximate narrow windows will inevitably produce skewed
        # results. This is also true of the first and last set of windows, regardless, especially
        # when sparse.
        builder.partition(w)
        builder.fit(predicate, self.model_type, regularize=regularize, resample=resample)
        model = builder.merge()
        model.calibrate_prediction_interval(alpha=self.calibration_alpha)
        return model

    def fit_model_with_revision(self, source, revision, regularize=False):
        masked_in_all_recs = mask_in(source, revision)
        aggregate = GlycoformAggregator(masked_in_all_recs)
        tags = {t.tag for t in revision}
        return self.fit_model(aggregate, lambda x: x.tag in tags, regularize=regularize)

    def revise_with(self, model, chromatograms, delta=0.65, min_time_difference=None):
        if min_time_difference is None:
            min_time_difference = max(model.width_range.lower, 0.5)
        reviser = IntervalModelReviser(
            model, [
                AmmoniumMaskedRule,
                AmmoniumUnmaskedRule,
                IsotopeRule,
                IsotopeRule2,
                HexNAc2Fuc1NeuAc2ToHex6AmmoniumRule
            ],
            chromatograms, valid_glycans=self.valid_glycans)
        reviser.evaluate()
        assert not np.isnan(model.width_range.lower)
        revised = reviser.revise(0.2, delta,  min_time_difference)
        self.log("... Revising Observations")
        k = 0
        local_aggregate = GlycoformAggregator(chromatograms)
        for new, old in zip(revised, chromatograms):
            if new.structure != old.structure:
                self.revised_tags.add(new.tag)
                neighbor_count = len(local_aggregate[old])
                k += 1
                self.log(
                    "...... %s: %0.2f %s (%0.2f) => %s (%0.2f) %0.2f/%0.2f with %d neighbors" % (
                        new.tag, new.apex_time,
                        old.glycan_composition, model.predict(old),
                        new.glycan_composition, model.predict(new),
                        model.score_interval(old, alpha=0.01),
                        model.score_interval(new, alpha=0.01),
                        neighbor_count - 1,
                    )
                )
        self.log("... Updated %d assignments" % (k, ))
        return revised

    def compute_model_coverage(self, model, aggregate, tag=True):
        coverages = []
        for i, chrom in enumerate(aggregate):
            if tag:
                chrom.annotations['tag'] = i
            coverage = model.coverage_for(chrom)

            coverages.append(coverage)
        return np.array(coverages)

    def subset_aggregate_by_coverage(self, chromatograms, coverages, threshold):
        # This happens when there are *no* models that span this observation's
        # time point. We force them to be included.
        nans = np.isnan(coverages)
        if nans.any():
            self.log("Encountered %d missing observations" % nans.sum())
        return np.array(list(chromatograms))[np.where((coverages >= threshold) | nans | np.isclose(coverages, threshold))[0]]

    def reweight(self, model, chromatograms, base_weight=0.3):
        reweighted = []
        # model_weight = 1.0 - base_weight
        total = 0.0
        for rec in chromatograms:
            rec = rec.copy()
            w = model.score_interval(rec, 0.01)
            if np.isnan(w):
                w = 0
            if w != 0:
                w **= -1
            rec.weight = w
            # rec.weight = base_weight + (w * model_weight)
            if np.isnan(rec.weight):
                rec.weight = base_weight
            reweighted.append(rec)
            total += w
        self.log("Total Score for %d elements = %f" % (len(chromatograms), total))
        return reweighted

    def make_groups_from(self, chromatograms):
        return GlycoformAggregator(chromatograms).by_peptide.values()

    def find_uncovered_group_members(self, chromatograms, coverages, min_size=2):
        recs = []
        for chrom, cov in zip(chromatograms, coverages):
            if not np.isclose(cov, 1.0) and len(self.aggregate[chrom]) > min_size:
                recs.append(chrom)
        return recs

    def run(self, k=10, base_weight=0.3, revision_threshold=0.35, final_round=True, regularize=True):
        self.log("Fitting first model...")
        # The first model is fit on those cases where we have both
        # the Unmodified form as well as a modified form e.g. Ammonium-adducted
        self.current_model = self.fit_first_model(regularize=regularize)

        all_records = list(self.aggregate)

        # Compute the coverage over all the chromatograms, where coverage is defined
        # as the sum of the weights of all models that span that chromatogram which
        # have a definition for that peptide divided by the sum of all model weights
        # that span that chromatogram, regardless of whether they contain the peptide
        coverages = self.compute_model_coverage(self.current_model, all_records)

        # Select only those chromatograms that have 100% coverage (always defined in the model)
        # but regardless of modification state.
        covered_recs = self.subset_aggregate_by_coverage(all_records, coverages, 1.0)

        # Attempt to revise chromatograms which are covered by the current model
        revised_recs = self.revise_with(
            self.current_model, covered_recs, 0.35, max(self.current_model.width_range.lower  * 0.5, 0.5))

        # Update the list of chromatogram records
        all_records = mask_in(list(self.aggregate), revised_recs)

        # Fit a new model on the revised data
        self.current_model = model = self.fit_model_with_revision(
            all_records, revised_recs, regularize=regularize)
        self.revision_history.append((revised_recs, self.revised_tags.copy()))

        delta_time_scale = 1.0
        minimum_delta = 2.0
        k_scale = 2.0

        # Record how many chromatograms were kept last time
        last_covered = covered_recs
        for i in range(k):
            self.log("Iteration %d" % i)
            coverages = self.compute_model_coverage(model, all_records)
            coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
            covered_recs = self.subset_aggregate_by_coverage(
                all_records, coverages, coverage_threshold)
            new = set()
            if len(covered_recs) == len(last_covered):
                self.log("No new observations added, skipping")
                continue
            else:
                new = {c.tag for c in covered_recs} - {c.tag for c in last_covered}
                last_covered = covered_recs
            self.log("... Covering %d chromatograms at threshold %0.2f" % (
                len(covered_recs), coverage_threshold))
            self.log("... Added new tags: %r" % sorted(new))
            revised_recs = self.revise_with(
                model, covered_recs, revision_threshold,
                max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))
            revised_recs = self.reweight(model, revised_recs, base_weight)
            all_records = mask_in(all_records, revised_recs)

            self.current_model = model = self.fit_model_with_revision(
                all_records, revised_recs, regularize=regularize)

            self.revision_history.append((revised_recs, self.revised_tags.copy()))

        if final_round:
            self.log("Open Update Round")
            coverages = self.compute_model_coverage(model, all_records)
            coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
            covered_recs = self.subset_aggregate_by_coverage(
                all_records, coverages, coverage_threshold)
            extra_recs = self.find_uncovered_group_members(all_records, coverages)

            self.log("... Added new tags: %r" % sorted({c.tag for c in extra_recs}))
            covered_recs = np.concatenate((covered_recs, extra_recs))
            covered_recs = self.reweight(model, covered_recs, base_weight=0.01)

            all_records = mask_in(all_records, covered_recs)
            self.current_model = model = self.fit_model_with_revision(
                all_records, covered_recs, regularize=regularize)
            revised_recs = self.revise_with(
                model, covered_recs, revision_threshold,
                max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))

            all_records = mask_in(all_records, revised_recs)

            for i in range(k):
                self.log("Iteration %d" % (i + k))
                coverages = self.compute_model_coverage(model, all_records)
                coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
                covered_recs = self.subset_aggregate_by_coverage(
                    all_records, coverages, coverage_threshold)
                new = set()
                if len(covered_recs) == len(last_covered):
                    self.log("No new observations added, skipping")
                    continue
                else:
                    new = {c.tag for c in covered_recs} - {c.tag for c in last_covered}
                    last_covered = covered_recs
                self.log("... Covering %d chromatograms at threshold %0.2f" % (
                    len(covered_recs), coverage_threshold))
                self.log("... Added new tags: %r" % sorted(new))
                revised_recs = self.revise_with(
                    model, covered_recs, revision_threshold,
                    max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))
                revised_recs = self.reweight(model, revised_recs, base_weight)
                all_records = mask_in(all_records, revised_recs)

                self.current_model = model = self.fit_model_with_revision(
                    all_records, revised_recs, regularize=regularize)

                self.revision_history.append((revised_recs, self.revised_tags.copy()))

            self.log("Final Update Round")
            coverages = self.compute_model_coverage(model, all_records)
            coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
            covered_recs = self.subset_aggregate_by_coverage(
                all_records, coverages, coverage_threshold)

            extra_recs = self.find_uncovered_group_members(all_records, coverages)

            self.log("... Added new tags: %r" % sorted({c.tag for c in extra_recs}))
            covered_recs = np.concatenate((covered_recs, extra_recs))
            covered_recs = self.reweight(model, covered_recs, base_weight=0.01)

            all_records = mask_in(all_records, covered_recs)
            self.current_model = model = self.fit_model_with_revision(
                all_records, covered_recs, regularize=regularize)
            revised_recs = self.revise_with(
                model, covered_recs, revision_threshold,
                max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))

            all_records = mask_in(all_records, revised_recs)


        return model, all_records







class ElutionTimeModel(ScoringFeatureBase):
    feature_type = 'elution_time'

    def __init__(self, fit=None, factors=None):
        self.fit = fit
        self.factors = factors

    def configure(self, analysis_data):
        if self.fit is None:
            matches = analysis_data['matches']
            fitter = AbundanceWeightedFactorElutionTimeFitter(
                matches, self.factors)
            fitter.fit()
            self.fit = fitter

    def score(self, chromatogram, *args, **kwargs):
        if self.fit is not None:
            return self.fit.score(chromatogram)
        else:
            return 0.5
