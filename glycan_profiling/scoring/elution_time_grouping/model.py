import csv

from numbers import Number
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np
from scipy import stats

from glycopeptidepy import PeptideSequence
from glypy.utils import make_counter

from glycan_profiling import chromatogram_tree

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycan_profiling.chromatogram_tree import Unmodified, Ammonium
from glycan_profiling.scoring.base import (
    ScoringFeatureBase,)

from .structure import _get_apex_time, GlycopeptideChromatogramProxy, GlycoformAggregator, DeltaOverTimeFilter
from .linear_regression import (ransac, weighted_linear_regression_fit, prediction_interval, SMALL_ERROR)


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


class AbundanceWeightedMixin(object):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            (x.total_signal * x.weight) for x in self.chromatograms
        ]
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
        return t + self.transform(chromatogram)

    def _prepare_data_vector(self, chromatogram):
        return np.array([1, chromatogram.weighted_neutral_mass, ])

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
        )).T

    def build_weight_matrix(self):
        return np.eye([x.weight for x in self.chromatograms])


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

    def _fit(self, resample=False):
        if resample:
            solution = ransac(self.data, self.apex_time_array,
                              self.weight_matrix)
            alt = weighted_linear_regression_fit(
                self.data, self.apex_time_array, self.weight_matrix)
            if alt.R2 > solution.R2:
                return alt
            return solution
        else:
            solution = weighted_linear_regression_fit(
                self.data, self.apex_time_array, self.weight_matrix)
        return solution

    def fit(self, resample=False):
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

    def __init__(self, chromatograms, scale=1, transform=None, width_range=None):
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
        self._init_model_data()

    def build_weight_matrix(self):
        return np.eye([c.weight for c in self.chromatograms])

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
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self.factors = list(factors)
        super(FactorElutionTimeFitter, self).__init__(
            chromatograms, scale=scale, transform=transform, width_range=width_range)

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


class PeptideGroupChromatogramFeatureizer(FactorChromatogramFeatureizer):

    def _get_peptide_key(self, chromatogram):
        return str(PeptideSequence(str(chromatogram.structure)).deglycosylate())

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
                peptide_key = self._get_peptide_key(chromatogram)
                peptides[indicator[peptide_key]] = 1
            except KeyError:
                import warnings
                warnings.warn(
                    "Peptide sequence of %s not part of the model." % (chromatogram, ))
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
            peptide = self._get_peptide_key(peptide)
        return peptide in self._peptide_to_indicator


class PeptideFactorElutionTimeFitter(PeptideGroupChromatogramFeatureizer, FactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self._peptide_to_indicator = defaultdict(make_counter(0))
        self.by_peptide = defaultdict(list)
        self.peptide_groups = []
        # Ensure that _peptide_to_indicator is properly initialized
        for obs in chromatograms:
            key = self._get_peptide_key(obs)
            self.peptide_groups.append(self._peptide_to_indicator[key])
            self.by_peptide[key].append(obs)
        self.peptide_groups = np.array(self.peptide_groups)

        super(PeptideFactorElutionTimeFitter, self).__init__(
            chromatograms, list(factors), scale=scale, transform=transform,
            width_range=width_range)

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


class ModelEnsemble(object):
    def __init__(self, models, width_range=None):
        self.models = models
        self._models = list(models.values())
        self.width_range = IntervalRange(width_range)
        self._chromatograms = None

    def _models_for(self, chromatogram):
        if not isinstance(chromatogram, Number):
            point = chromatogram.apex_time
        else:
            point = chromatogram
        for model in self._models:
            if model.contains(point):
                weight = abs(model.centroid - point) + 1
                yield model, 1.0 / weight

    def predict_interval(self, chromatogram, alpha=0.05, merge=True):
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            preds.append(mod.predict_interval(chromatogram, alpha=alpha))
            weights.append(w)
        weights = np.array(weights)
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

    def predict(self, chromatogram, merge=True):
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            preds.append(mod.predict(chromatogram))
            weights.append(w)
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
            for chrom in model.chromatograms:
                chroma.add(chrom)
        self._chromatograms = sorted(chroma, key=lambda x: x.apex_time)
        return self._chromatograms

    @property
    def chromatograms(self):
        return self._get_chromatograms()


def unmodified_modified_predicate(x):
    return Unmodified in x.mass_shifts and Ammonium in x.mass_shifts


class EnsembleBuilder(object):
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

    def estimate_delta_widths(self):
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
            group_widths[t] = best
        return group_widths

    def fit(self, predicate=None, model_type=None):
        if model_type is None:
            from glycan_profiling.scoring.elution_time_grouping.model import AbundanceWeightedPeptideFactorElutionTimeFitter
            model_type = AbundanceWeightedPeptideFactorElutionTimeFitter
        if predicate is None:
            predicate = bool
        models = OrderedDict()
        for point, members in self.centers.items():
            obs = list(filter(bool, members))
            m = models[point] = model_type(obs, self.aggregator.factors)
            m.fit()
        self.models = models
        return self

    def merge(self):
        return ModelEnsemble(self.models)


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
