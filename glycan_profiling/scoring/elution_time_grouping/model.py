import csv
from collections import defaultdict

import numpy as np
from scipy import stats

from glycopeptidepy import PeptideSequence
from glypy.utils import make_counter

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


from glycan_profiling.scoring.base import (
    ScoringFeatureBase,)

from .structure import _get_apex_time, GlycopeptideChromatogramProxy
from .linear_regression import (ransac, weighted_linear_regression_fit, prediction_interval, SMALL_ERROR)



class ElutionTimeFitter(ScoringFeatureBase):
    feature_type = 'elution_time'

    def __init__(self, chromatograms, scale=1):
        self.chromatograms = chromatograms
        self.neutral_mass_array = None
        self.data = None
        self.apex_time_array = None
        self.weight_matrix = None
        self.parameters = None
        self.residuals = None
        self.estimate = None
        self.scale = scale
        self._init_model_data()

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

    def _get_apex_time(self, chromatogram):
        return _get_apex_time(chromatogram)

    def build_weight_matrix(self):
        return np.eye(len(self.chromatograms))

    def _prepare_data_vector(self, chromatogram):
        return np.array([1, chromatogram.weighted_neutral_mass,])

    def feature_names(self):
        return ['intercept', 'mass']

    def _prepare_data_matrix(self, mass_array):
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
        )).T

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

    def _df(self):
        return max(len(self.chromatograms) - len(self.parameters), 1)

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
        super(FactorElutionTimeFitter, self).__init__(
            chromatograms, scale=scale)

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
        super(PeptideFactorElutionTimeFitter, self).__init__(
            chromatograms, list(factors), scale)

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
                    "Peptide sequence %s not part of the model." % (peptide_key, ))
        return np.array(
            peptides + [chromatogram.glycan_composition[f] for f in self.factors])


class AbundanceWeightedPeptideFactorElutionTimeFitter(PeptideFactorElutionTimeFitter):
    def build_weight_matrix(self):
        W = np.eye(len(self.chromatograms)) * [
            (x.total_signal) for x in self.chromatograms
        ]
        W /= W.max()
        return W

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
