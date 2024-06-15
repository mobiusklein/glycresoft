import csv
import itertools
import logging
import gzip
import io
import array

from numbers import Number
from collections import defaultdict, namedtuple
from functools import partial
from typing import (Any, Callable, ClassVar,
                    DefaultDict, Dict, Iterator,
                    List, Optional, Set, Tuple,
                    Type, Union, OrderedDict)
from multiprocessing import cpu_count
from concurrent import futures

import numpy as np
from scipy import stats

import dill

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from glypy.utils import make_counter
from glypy.structure.glycan_composition import HashableGlycanComposition, FrozenMonosaccharideResidue

from glycopeptidepy.structure.sequence.implementation import PeptideSequence
from glycresoft.chromatogram_tree.chromatogram import Chromatogram, ChromatogramInterface

from glycresoft.database.composition_network.space import composition_distance, DistanceCache


from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycresoft.task import TaskBase

from glycresoft.chromatogram_tree import Unmodified, Ammonium
from glycresoft.scoring.base import ScoringFeatureBase

from glycresoft.structure import FragmentCachingGlycopeptide
from glycresoft.structure.structure_loader import GlycanCompositionDeltaCache

from .structure import (
    ChromatogramProxy, _get_apex_time, GlycopeptideChromatogramProxy,
    GlycoformAggregator, DeltaOverTimeFilter)
from .linear_regression import (
    WLSSolution, ransac, weighted_linear_regression_fit,
    prediction_interval, SMALL_ERROR,
    weighted_linear_regression_fit_ridge)
from .reviser import (IntervalModelReviser, IsotopeRule, AmmoniumMaskedRule,
                      AmmoniumUnmaskedRule, HexNAc2NeuAc2ToHex6AmmoniumRule,
                      IsotopeRule2, HexNAc2Fuc1NeuAc2ToHex7, PhosphateToSulfateRule,
                      SulfateToPhosphateRule, Sulfate1HexNAc2ToHex3Rule, Hex3ToSulfate1HexNAc2Rule,
                      Phosphate1HexNAc2ToHex3Rule, Hex3ToPhosphate1HexNAc2Rule, HexNAc2NeuAc2ToHex6Deoxy,
                      AmmoniumMaskedNeuGcRule, AmmoniumUnmaskedNeuGcRule, IsotopeRuleNeuGc,
                      RevisionValidatorBase,
                      PeptideYUtilizationPreservingRevisionValidator,
                      RevisionRuleList,
                      RuleBasedFDREstimator,
                      ResidualFDREstimator,
                      ValidatedGlycome)

from . import reviser as libreviser

logger = logging.getLogger("glycresoft.elution_time_model")
logger.addHandler(logging.NullHandler())


CALIBRATION_QUANTILES = [0.25, 0.75]


ChromatogramType = Union[Chromatogram, ChromatogramProxy, ChromatogramInterface]


class IntervalRange(object):
    lower: float
    upper: float

    def __init__(self, lower=None, upper=None):
        if isinstance(lower, IntervalRange):
            self.lower = lower.lower
            self.upper = lower.upper
        elif isinstance(lower, (tuple, list)):
            self.lower, self.upper = lower
        else:
            self.lower = lower
            self.upper = upper

    def clamp(self, value: float) -> float:
        if self.lower is None:
            return value
        if value < self.lower:
            return self.lower
        if value > self.upper:
            return self.upper
        return value

    def interval(self, value: List[float]) -> List[float]:
        center = np.mean(value)
        lower = center - self.clamp(abs(center - value[0]))
        upper = center + self.clamp(abs(value[1] - center))
        return [lower, upper]

    def __repr__(self):
        return "{self.__class__.__name__}({self.lower}, {self.upper})".format(self=self)


class AbundanceWeightedMixin(object):
    def build_weight_matrix(self) -> np.ndarray:
        W = np.array([
            1.0 / (x.total_signal * x.weight) for x in self.chromatograms
        ])
        if len(self.chromatograms) == 0:
            return np.diag(W)
        W /= W.max()
        return W


class ChromatgramFeatureizerBase(object):

    transform = None

    def feature_names(self) -> List[str]:
        return ['intercept', 'mass']

    def _get_apex_time(self, chromatogram: ChromatogramType) -> float:
        t = _get_apex_time(chromatogram)
        if self.transform is None:
            return t
        return t - self.transform(chromatogram)

    def _prepare_data_vector(self, chromatogram: ChromatogramType) -> np.ndarray:
        return np.array([1, chromatogram.weighted_neutral_mass, ])

    def _prepare_data_matrix(self, mass_array, chromatograms: Optional[List[ChromatogramType]]=None) -> np.ndarray:
        if chromatograms is None:
            chromatograms = self.chromatograms
        return np.vstack((
            np.ones(len(mass_array)),
            np.array(mass_array),
        )).T

    def build_weight_matrix(self) -> np.ndarray:
        return 1.0 / np.array([x.weight for x in self.chromatograms])


class PredictorBase(object):
    transform = None

    def predict(self, chromatogram: ChromatogramType) -> float:
        t = self._predict(self._prepare_data_vector(chromatogram))
        if self.transform is None:
            return t
        return t + self.transform(chromatogram)

    def _predict(self, x: np.ndarray) -> float:
        return x.dot(self.parameters)

    def predict_interval(self, chromatogram: ChromatogramType, alpha: float = 0.05) -> np.ndarray:
        x = self._prepare_data_vector(chromatogram)
        return self._predict_interval(x, alpha=alpha)

    def _predict_interval(self, x: np.ndarray, alpha: float=0.05) -> np.ndarray:
        y = self._predict(x)
        return prediction_interval(self.solution, x, y, alpha=alpha)

    def __call__(self, x: ChromatogramType) -> float:
        return self.predict(x)


class LinearModelBase(PredictorBase, SpanningMixin):
    def _init_model_data(self):
        self.neutral_mass_array = np.array([
            x.weighted_neutral_mass for x in self.chromatograms
        ])
        self.data = self._prepare_data_matrix(self.neutral_mass_array, self.chromatograms)

        self.apex_time_array = np.array([
            self._get_apex_time(x) for x in self.chromatograms
        ])

        self.weight_matrix = self.build_weight_matrix()
        self._update_model_time_range()

    def _update_model_time_range(self):
        if len(self.apex_time_array) == 0:
            self.start = 0.0
            self.end = 0.0
            self.centroid = 0.0
        else:
            self.start = self.apex_time_array.min()
            self.end = self.apex_time_array.max()
            if self.weight_matrix.ndim > 1:
                d = np.diag(self.weight_matrix)
            else:
                d = self.weight_matrix
            self.centroid = self.apex_time_array.dot(d) / d.sum()

    @property
    def start_time(self) -> float:
        return self.start

    @property
    def end_time(self) -> float:
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

    def default_regularization(self) -> np.ndarray:
        p = self.data.shape[1]
        return np.ones_like(p) * 0.001

    @property
    def estimate(self) -> np.ndarray:
        if self.solution is None:
            return None
        return self.solution.yhat

    @estimate.setter
    def estimate(self, value):
        pass

    @property
    def residuals(self) -> np.ndarray:
        if self.solution is None:
            return None
        return self.solution.residuals

    @residuals.setter
    def residuals(self, value):
        pass

    @property
    def parameters(self) -> np.ndarray:
        if self.solution is None:
            return None
        return self.solution.parameters

    @parameters.setter
    def parameters(self, value):
        pass

    @property
    def projection_matrix(self) -> np.ndarray:
        if self.solution is None:
            return None
        return self.solution.projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        pass

    def fit(self, resample=False, alpha=None):
        solution = self._fit(resample=resample, alpha=alpha)
        self.solution = solution
        return self

    def loglikelihood(self) -> float:
        n = self.data.shape[0]
        n2 = n / 2.0
        rss = self.solution.rss
        # The "concentrated likelihood"
        likelihood = -np.log(rss) * n2
        # The likelihood constant
        likelihood -= (1 + np.log(np.pi / n2)) / n2
        if self.weight_matrix.ndim > 1:
            W = np.diag(self.weight_matrix)
        else:
            W = self.weight_matrix
        likelihood += 0.5 * np.sum(np.log(W))
        return likelihood

    @property
    def rss(self) -> float:
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        if w.ndim > 1:
            w = np.diag(w)
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss = (w * residuals * residuals).sum()
        return rss

    @property
    def mse(self) -> float:
        return self.rss / (len(self.apex_time_array) - len(self.parameters) - 1.0)

    def parameter_significance(self) -> np.ndarray:
        if self.weight_matrix.ndim > 1:
            W = np.diag(self.weight_matrix)
        else:
            W = self.weight_matrix
        XtWX_inv = np.linalg.pinv(
            ((self.data.T * W).dot(self.data)))
        # With unknown variance, use the mean squared error estimate
        sigma_params = np.sqrt(np.diag(self.mse * XtWX_inv))
        degrees_of_freedom = len(self.apex_time_array) - \
            len(self.parameters) - 1.0
        # interval = stats.t.interval(1 - alpha / 2.0, degrees_of_freedom)
        t_score = np.abs(self.parameters) / sigma_params
        p_value = stats.t.sf(t_score, degrees_of_freedom) * 2
        return p_value

    def parameter_confidence_interval(self, alpha=0.05) -> np.ndarray:
        if self.weight_matrix.ndim > 1:
            W = np.diag(self.weight_matrix)
        else:
            W = self.weight_matrix
        X = self.data
        sigma_params = np.sqrt(
            np.diag((self.mse) * np.linalg.pinv(
                (X.T * W).dot(X))))
        degrees_of_freedom = len(self.apex_time_array) - \
            len(self.parameters) - 1
        iv = stats.t.interval((1 - alpha) / 2., degrees_of_freedom)
        iv = np.array(iv) * sigma_params.reshape((-1, 1))
        return np.array(self.parameters).reshape((-1, 1)) + iv

    def R2(self, adjust=True) -> float:
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        if w.ndim > 1:
            w = np.diag(w)
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss = (w * residuals * residuals).sum()
        tss = (y - y.mean())
        tss = (w * tss * tss).sum()
        n = len(y)
        k = len(self.parameters)
        if adjust:
            adjustment_factor = (n - 1.0) / max(float(n - k - 1.0), 1)
        else:
            adjustment_factor = 1.0
        R2 = (1 - adjustment_factor * (rss / tss))
        return R2

    def _df(self) -> int:
        return max(len(self.apex_time_array) - len(self.parameters), 1)


class IntervalScoringMixin(object):
    _interval_padding = 0.0

    def _threshold_interval(self, interval) -> np.ndarray:
        width = (interval[1] - interval[0]) / 2.0
        if self.width_range is not None:
            if np.isnan(width):
                width = self.width_range.upper
            else:
                width = self.width_range.clamp(width)
        return width

    def has_interval_been_thresholded(self, chromatogram: ChromatogramType, alpha: float = 0.05) -> bool:
        interval = self.predict_interval(chromatogram, alpha=alpha)
        width = (interval[1] - interval[0]) / 2.0
        thresholded_width = self._threshold_interval(interval)
        return width > thresholded_width

    def score_interval(self, chromatogram: ChromatogramType, alpha: float = 0.05) -> float:
        interval = self.predict_interval(chromatogram, alpha=alpha)
        pred = interval.mean()
        delta = abs(chromatogram.apex_time - pred)
        width = self._threshold_interval(interval) + self._interval_padding
        return max(1 - delta / width, 0.0)

    def _truncate_interval(self, interval):
        centroid = interval.mean()
        width = self._threshold_interval(interval) + self._interval_padding
        return centroid + np.array([-width, width])

    def calibrate_prediction_interval(self, chromatograms=None, alpha: float=0.05):
        if chromatograms is None:
            chromatograms = self.chromatograms
        ivs = np.array([self.predict_interval(c, alpha)
                        for c in chromatograms])
        widths = (ivs[:, 1] - ivs[:, 0]) / 2.0
        widths = widths[~np.isnan(widths)]
        self.width_range = IntervalRange(
            *np.quantile(widths, CALIBRATION_QUANTILES))
        if np.isnan(self.width_range.lower):
            raise ValueError("Width range cannot be NaN")
        return self

    @property
    def interval_padding(self) -> Optional[float]:
        return self._interval_padding

    @interval_padding.setter
    def interval_padding(self, value):
        if value is None:
            value = 0.0
        self._interval_padding = value


class ElutionTimeFitter(LinearModelBase, ChromatgramFeatureizerBase, ScoringFeatureBase, IntervalScoringMixin):
    feature_type = 'elution_time'

    chromatograms: Optional[List]
    neutral_mass_array: np.ndarray
    data: np.ndarray
    apex_time_array: np.ndarray
    weight_matrix: np.ndarray
    solution: WLSSolution
    scale: float
    transform: Any
    width_range: IntervalRange
    regularize: bool

    chromatogram_type: ClassVar[Type] = ChromatogramProxy

    def __init__(self, chromatograms, scale=1, transform=None, width_range=None, regularize=False):
        self.chromatograms = chromatograms
        self.neutral_mass_array = None
        self.data = None
        self.apex_time_array = None
        self.weight_matrix = None
        self.solution = None
        self.scale = scale
        self.transform = transform
        self.width_range = IntervalRange(width_range)
        self.regularize = regularize
        self._init_model_data()

    def __getstate__(self):
        state = {}
        state['chromatograms'] = self.chromatograms
        state['solution'] = self.solution
        state['scale'] = self.scale
        state['transform'] = self.transform
        state['width_range'] = self.width_range
        state['regularize'] = self.regularize
        state['start_time'] = self.start
        state['end_time'] = self.end
        state['centroid'] = self.centroid
        return state

    def __setstate__(self, state):
        self.chromatograms = state['chromatograms']
        self.solution = state['solution']
        self.scale = state['scale']
        self.transform = state['transform']
        self.width_range = state['width_range']
        self.regularize = state['regularize']
        self.start = state['start_time']
        self.end = state['end_time']
        self.centroid = state['centroid']

        if self.solution is not None:
            (self.data, self.apex_time_array) = self.solution.data
            self.weight_matrix = self.solution.weights
        if self.chromatograms:
            self._init_model_data()

    def __reduce__(self):
        return self.__class__, (self.chromatograms or [], ), self.__getstate__()

    def _get_chromatograms(self):
        return self.chromatograms

    def drop_chromatograms(self):
        self.chromatograms = None
        return self

    def score(self, chromatogram: ChromatogramType) -> float:
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
        sizes = list(map(max, zip(sizes, value_sizes)))
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

    def to_dict(self) -> Dict:
        out = OrderedDict()
        keys = self.feature_names()
        for k, v in zip(keys, self.parameters):
            out[k] = v
        return out

    def _infer_factors(self) -> List[str]:
        keys = set()
        for record in self.chromatograms:
            keys.update(record.glycan_composition)
        keys = sorted(map(str, keys))
        return keys

    def plot_residuals(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        w = np.diag(self.weight_matrix) if self.weight_matrix.ndim > 1 else self.weight_matrix
        ax.scatter(self.apex_time_array, self.residuals, s=w * 1000.0, alpha=0.25,
                   edgecolor='black')
        return ax


class SimpleLinearFitter(LinearModelBase):
    x: np.ndarray
    y: np.ndarray

    weight_matrix: np.ndarray
    data: np.ndarray

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = self.apex_time_array = np.array(y)
        self.weight_matrix = np.diag(np.ones_like(y))
        self.data = np.stack((np.ones_like(x), x)).T

    def predict(self, vx) -> np.ndarray:
        if not isinstance(vx, (list, tuple, np.ndarray)):
            vx = [vx]
        return np.stack((np.ones_like(vx), vx)).T.dot(self.parameters)

    def _prepare_data_vector(self, x) -> np.ndarray:
        return np.array([1., x])


class AbundanceWeightedElutionTimeFitter(AbundanceWeightedMixin, ElutionTimeFitter):
    pass


class FactorChromatogramFeatureizer(ChromatgramFeatureizerBase):
    def feature_names(self) -> List[str]:
        return ['intercept'] + self.factors

    def _prepare_data_matrix(self, mass_array, chromatograms: Optional[List[ChromatogramType]]=None) -> np.ndarray:
        if chromatograms is None:
            chromatograms = self.chromatograms
        return np.vstack([np.ones(len(mass_array)), ] + [
            np.array([c.glycan_composition[f] for c in chromatograms])
            for f in self.factors]).T

    def _prepare_data_vector(self, chromatogram: ChromatogramType, no_intercept=False) -> np.ndarray:
        intercept = 0 if no_intercept else 1
        return np.array(
            [intercept] + [
                chromatogram.glycan_composition[f] for f in self.factors])


class FactorTransform(PredictorBase, FactorChromatogramFeatureizer):
    factors: List[str]
    parameters: np.ndarray

    def __init__(self, factors, parameters, intercept=0.0):
        self.factors = factors
        # Add a zero intercept
        self.parameters = np.concatenate([[intercept], parameters])


class FactorElutionTimeFitter(FactorChromatogramFeatureizer, ElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None, regularize=False):
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self.factors = list(factors)
        self._coerced_factors = [
            FrozenMonosaccharideResidue.from_iupac_lite(f)
            for f in self.factors
        ]
        super(FactorElutionTimeFitter, self).__init__(
            chromatograms, scale=scale, transform=transform,
            width_range=width_range, regularize=regularize)

    def __getstate__(self):
        state = super(FactorElutionTimeFitter, self).__getstate__()
        state['factors'] = self.factors
        return state

    def __setstate__(self, state):
        super(FactorElutionTimeFitter, self).__setstate__(state)
        self.factors = state['factors']
        self._coerced_factors = [
            FrozenMonosaccharideResidue.from_iupac_lite(f)
            for f in self.factors
        ]

    def predict_delta_glycan(self, chromatogram: ChromatogramType, delta_glycan: HashableGlycanComposition) -> float:
        try:
            shifted = chromatogram.shift_glycan_composition(delta_glycan)
        except AttributeError:
            shifted = GlycopeptideChromatogramProxy.from_chromatogram(
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
        from glycresoft.plotting.colors import ColorMapper
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

    def predict(self, chromatogram: ChromatogramType, no_intercept=False) -> float:
        return self._predict(self._prepare_data_vector(chromatogram, no_intercept=no_intercept))


class AbundanceWeightedFactorElutionTimeFitter(AbundanceWeightedMixin, FactorElutionTimeFitter):
    pass


class PeptideBackboneKeyedMixin(object):
    def get_peptide_key(self, chromatogram: ChromatogramType) -> str:
        try:
            return chromatogram.peptide_key
        except AttributeError:
            return GlycopeptideChromatogramProxy.from_chromatogram(chromatogram).peptide_key


class PeptideGroupChromatogramFeatureizer(FactorChromatogramFeatureizer, PeptideBackboneKeyedMixin):

    def _prepare_data_matrix(self, mass_array: np.ndarray,
                             chromatograms: Optional[List[ChromatogramType]]=None) -> np.ndarray:
        if chromatograms is None:
            chromatograms = self.chromatograms
        p = len(self._peptide_to_indicator)
        n = len(chromatograms)
        peptides = np.zeros((n, p + len(self.factors)))
        for i, c in enumerate(chromatograms):
            peptide_key = self.get_peptide_key(c)
            if peptide_key in self._peptide_to_indicator:
                j = self._peptide_to_indicator[peptide_key]
                peptides[i, j] = 1
            gc = c.glycan_composition
            for j, f in enumerate(self._coerced_factors, p):
                peptides[i, j] = gc._getitem_fast(f)

        # Omit the intercept, so that all peptide levels are used without inducing linear dependence.
        return peptides

    def feature_names(self) -> List[str]:
        names = []
        peptides = [None] * len(self._peptide_to_indicator)
        for key, value in self._peptide_to_indicator.items():
            peptides[value] = key
        names.extend(peptides)
        names.extend(self.factors)
        return names

    def _prepare_data_vector(self, chromatogram: ChromatogramType, no_intercept=False) -> np.ndarray:
        k = len(self._peptide_to_indicator)
        feature_vector = np.zeros(k + len(self.factors))
        max_peptide_i = k - 1
        if not no_intercept:
            peptide_key = self.get_peptide_key(chromatogram)
            if peptide_key in self._peptide_to_indicator:
                feature_vector[self._peptide_to_indicator[peptide_key]] = 1
            else:
                logger.debug(
                    "Peptide sequence of %s not part of the model.", chromatogram)
        gc = chromatogram.glycan_composition
        for i, f in enumerate(self._coerced_factors, 1 + max_peptide_i):
            feature_vector[i] = gc._getitem_fast(f)
        return feature_vector

    def has_peptide(self, peptide) -> bool:
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

    def predict_component_times(self, chromatogram: ChromatogramType):
        y_gp = self.predict(chromatogram)
        deglyco = chromatogram.copy()
        deglyco.glycan_composition.clear()
        y_p = self.predict(deglyco)
        y_g = y_gp - y_p
        return y_p, y_g


class PeptideFactorElutionTimeFitter(PeptideGroupChromatogramFeatureizer, FactorElutionTimeFitter):
    _peptide_to_indicator: DefaultDict[str, int]
    by_peptide: DefaultDict[str, List]
    peptide_groups: List

    chromatogram_type: ClassVar[Type] = GlycopeptideChromatogramProxy

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

    def __getstate__(self):
        state = super(PeptideFactorElutionTimeFitter, self).__getstate__()
        state['_peptide_to_indicator'] = self._peptide_to_indicator
        state['peptide_groups'] = self.peptide_groups
        state['by_peptide'] = self.by_peptide
        return state

    def __setstate__(self, state):
        super(PeptideFactorElutionTimeFitter, self).__setstate__(state)
        self._peptide_to_indicator = state['_peptide_to_indicator']
        self.peptide_groups = state['peptide_groups']
        self.by_peptide = state['by_peptide']


    def drop_chromatograms(self):
        super(PeptideFactorElutionTimeFitter, self).drop_chromatograms()
        self.by_peptide = {k: [] for k in self.by_peptide}
        return self

    @property
    def peptide_count(self) -> int:
        return len(self.by_peptide)

    def default_regularization(self) -> np.ndarray:
        monosaccharide_alphas = {}
        alpha = np.concatenate((
            np.zeros(self.peptide_count),
            [monosaccharide_alphas.get(f, 0.01) for f in self.factors]
        ))
        return alpha

    def fit(self, resample=False, alpha=None):
        if alpha is None and self.regularize:
            alpha = self.default_regularization()
        return super().fit(resample, alpha)

    def groupwise_R2(self, adjust=True):
        x = self.data
        y = self.apex_time_array
        w = self.weight_matrix
        if w.ndim > 1:
            w = np.diag(w)
        yhat = x.dot(self.parameters)
        residuals = (y - yhat)
        rss_u = (w * residuals * residuals)
        tss = (y - y.mean())
        tss_u = (w * tss * tss)

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
        if self.weight_matrix.ndim > 1:
            weights = np.diag(self.weight_matrix)
        else:
            weights = self.weight_matrix
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
        if self.weight_matrix.ndim > 1:
            weights = np.diag(self.weight_matrix)
        else:
            weights = self.weight_matrix
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


class ModelEnsemble(PeptideBackboneKeyedMixin, IntervalScoringMixin):
    models: OrderedDict[float, ElutionTimeFitter]
    _models: List[ElutionTimeFitter]
    _chromatograms: Optional[List[GlycopeptideChromatogramProxy]]
    _summary_statistics: Dict[str, Any]
    width_range: IntervalRange
    residual_fdr: Optional[ResidualFDREstimator]

    def __init__(self, models, width_range=None):
        self._set_models(models)
        self.width_range = IntervalRange(width_range)
        self._chromatograms = None
        self.external_peptide_offsets = {}
        self._summary_statistics = {}
        self.residual_fdr = None

    @property
    def model_width(self) -> float:
        points = sorted(self.models)
        return np.mean(np.diff(points)) * 4

    def __reduce__(self):
        return self.__class__, (OrderedDict(), self.width_range), self.__getstate__()

    def __getstate__(self):
        state = dict()
        models = OrderedDict()
        for key, value in self.models.items():
            buffer = io.BytesIO()
            writer = gzip.GzipFile(fileobj=buffer, mode='wb')
            dill.dump(value, writer, 2)
            writer.flush()
            models[key] = buffer.getvalue()
        state['models'] = models
        state['summary_statistics'] = self._summary_statistics
        state['interval_padding'] = self._interval_padding
        state['residual_fdr'] = self.residual_fdr
        return state

    def __setstate__(self, state):
        if not state:
            return
        models = OrderedDict()
        for key, buffer in  state.get('models', {}).items():
            buffer = io.BytesIO(buffer)
            reader = gzip.GzipFile(fileobj=buffer)
            model = dill.load(reader)
            models[key] = model
        if models:
            self._set_models(models)
        self._summary_statistics = state.get("summary_statistics", {})
        self._interval_padding = state.get("interval_padding", 0.0)
        self.residual_fdr = state.get('residual_fdr')

    def _set_models(self, models):
        self.models = models
        self._models = list(models.values())

    def _compute_summary_statistics(self) -> Dict[str, Any]:
        chromatograms = self.chromatograms
        apex_time_array = np.array([
            c.apex_time for c in chromatograms
        ])
        labels = [(str(c.structure), c.mass_shifts) for c in chromatograms]
        predicted_apex_time_array = np.array([
            self.predict(c) for c in chromatograms
        ])
        confidence_intervals = np.array([
            self.predict_interval(c, alpha=0.01) for c in chromatograms
        ])
        residuals_array = apex_time_array - predicted_apex_time_array
        self._summary_statistics.update({
            "apex_time_array": apex_time_array,
            "predicted_apex_time_array": predicted_apex_time_array,
            "confidence_intervals": confidence_intervals,
            "residuals_array": residuals_array,
            "labels": labels,
            "original_width_range": [self.width_range.lower, self.width_range.upper] if self.width_range else None
        })

    def R2(self, adjust=True) -> float:
        apex_time_array = self._summary_statistics['apex_time_array']
        predicted_apex_time_array = self._summary_statistics['predicted_apex_time_array']
        mask = ~np.isnan(predicted_apex_time_array)
        return np.corrcoef(apex_time_array[mask], predicted_apex_time_array[mask])[0, 1] ** 2

    def _models_for(self, chromatogram: Union[Number, ChromatogramType]) -> Iterator[Tuple[ElutionTimeFitter, float]]:
        if not isinstance(chromatogram, Number):
            point = chromatogram.apex_time
        else:
            point = chromatogram
        for model in self._models:
            if model.contains(point):
                weight = abs(model.centroid - point) + 1
                yield model, 1.0 / weight

    def models_for(self, chromatogram: Union[Number, ChromatogramType]) -> List[Tuple[ElutionTimeFitter, float]]:
        return list(self._models_for(chromatogram))

    def coverage_for(self, chromatogram: GlycopeptideChromatogramProxy) -> float:
        refs = array.array('d')
        weights = array.array('d')
        for mod, weight in self._models_for(chromatogram):
            refs.append(mod.has_peptide(chromatogram))
            weights.append(weight)
        if len(weights) == 0:
            return 0
        else:
            coverage = np.dot(refs, weights) / np.sum(weights)
        return coverage

    def has_peptide(self, chromatogram: ChromatogramType) -> bool:
        return any(m.has_peptide(chromatogram) for m in self._models)

    def estimate_query_point(self,
                             glycopeptide: Union[str, PeptideSequence, ChromatogramType]
                            ) -> GlycopeptideChromatogramProxy:
        if isinstance(glycopeptide, str):
            key = str(PeptideSequence(glycopeptide).deglycosylate())
            glycopeptide = PeptideSequence(glycopeptide)
            case = GlycopeptideChromatogramProxy(
                glycopeptide.total_mass,
                -1, 1, glycopeptide.glycan_composition, structure=glycopeptide)
        elif isinstance(glycopeptide, PeptideSequence):
            key = str(glycopeptide.clone().deglycosylate())
            case = GlycopeptideChromatogramProxy(
                glycopeptide.total_mass,
                -1, 1, glycopeptide.glycan_composition, structure=glycopeptide)
        else:
            key = self.get_peptide_key(glycopeptide)
            case = glycopeptide.clone()
        preds = []
        for t, mod in self.models.items():
            if mod.has_peptide(key):
                preds.append(mod.predict(case))
        query_point = np.nanmedian(preds)
        case.apex_time = query_point
        return case

    def predict_interval_external_peptide(self, chromatogram: ChromatogramType, alpha=0.05, merge=True) -> List[float]:
        key = self.get_peptide_key(chromatogram)
        offset = self.external_peptide_offsets[key]
        interval = self.predict_interval(chromatogram, alpha=alpha, merge=merge)
        if merge:
            interval += offset
        else:
            interval = [iv + offset for iv in interval]
        return interval

    def predict_interval(self, chromatogram: ChromatogramType, alpha: float=0.05,
                         merge: bool=True, check_peptide: bool=True) -> np.ndarray:
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
        if len(weights) == 0:
            return np.array([np.nan, np.nan])
        weights /= weights.max()
        if merge:
            if len(weights) == 0:
                return np.array([np.nan, np.nan])
            avg = np.average(preds, weights=weights, axis=0)
            return avg
        return preds, weights

    def predict(self, chromatogram: ChromatogramType, merge: bool=True,
                check_peptide: bool=True) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        weights = []
        preds = []
        for mod, w in self._models_for(chromatogram):
            if check_peptide and not mod.has_peptide(chromatogram):
                continue
            preds.append(mod.predict(chromatogram))
            weights.append(w)
        if len(weights) == 0:
            return np.nan
        weights = np.array(weights)
        weights /= weights.max()
        if merge:
            avg = np.average(preds, weights=weights)
            return avg
        return np.array(preds), weights

    def score(self, chromatogram: ChromatogramType, merge=True) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
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

    def drop_chromatograms(self):
        self._chromatograms = None
        for model in self._models:
            model.drop_chromatograms()
        return self

    @property
    def chromatograms(self):
        return self._get_chromatograms()

    def calibrate_prediction_interval(self, chromatograms: Optional[List[GlycopeptideChromatogramProxy]]=None,
                                      alpha: float=0.01):
        if chromatograms is None:
            chromatograms = self.chromatograms
        ivs = np.array([self.predict_interval(c, alpha)
                        for c in chromatograms])
        if len(ivs) == 0:
            widths = []
        else:
            widths = (ivs[:, 1] - ivs[:, 0]) / 2.0
            widths = widths[~np.isnan(widths)]
        if len(widths) == 0:
            self.width_range = IntervalRange(0.0, float('inf'))
        else:
            self.width_range = IntervalRange(*np.quantile(widths, CALIBRATION_QUANTILES))
            if np.isnan(self.width_range.lower):
                raise ValueError("Width range cannot be NaN")
        return self

    def plot_number_of_training_points(self, ax=None, color='teal', markerfacecolor='mediumspringgreen',
                                       markeredgecolor='mediumaquamarine', markersize=7, marker='.'):
        if ax is None:
            _fig, ax = plt.subplots(1)
        ts, n_pts = zip(*[(k, v.data.shape[0]) for k, v in self.models.items()])
        ax.plot(ts, n_pts, marker=marker, markersize=markersize,
                color=color,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)

        def nearest_multiple_of_ten(value):
            return round(value / 10) * 10

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        tick_multiple = (nearest_multiple_of_ten(max(n_pts)) // 3)
        if tick_multiple <= 0:
            tick_multiple = 5

        ax.yaxis.set_major_locator(plt.MultipleLocator(tick_multiple))
        ax.set_ylabel("# of Data Points", size=14)
        ax.set_xlabel("Time", size=14)
        return ax

    def plot_factor_coefficients(self, ax=None, weight_by_obs=False):
        if ax is None:
            _fig, ax = plt.subplots(1)
        from glycresoft.plotting.colors import get_color

        param_cis = {}

        def local_point(point, factor):
            val = []
            ci = []
            weights = []
            for mod, weight in self._models_for(point):
                i = mod.feature_names().index(factor)
                val.append(mod.parameters[i])
                try:
                    param_ci = param_cis[mod.start_time, mod.end_time]
                except KeyError:
                    param_cis[mod.start_time, mod.end_time] = param_ci = mod.parameter_confidence_interval()
                ci_i = param_ci[i]
                d = (ci_i[1] - ci_i[0]) / 2
                ci.append(d)
                weights.append(weight * len(mod.apex_time_array) if weight_by_obs else weight)
            if not weights:
                return 0.0, 0.0
            return np.average(val, weights=weights), np.average(ci, weights=weights)

        times = np.arange(self._models[0].start_time, self._models[-1].end_time, 1)
        factors = set()
        for x in self.models.values():
            factors.update(x.factors)
        for factor in factors:
            yval = []
            xval = []
            ci_width = []
            for t in times:
                xval.append(t)
                y, ci_delta = local_point(t, factor)
                yval.append(y)
                ci_width.append(ci_delta)
            c = get_color(factor)
            xval = np.array(xval)
            yval = np.array(yval)
            ci_width = np.array(ci_width)
            ax.plot(xval, yval, label=factor, marker='.', color=c)
            ax.fill_between(xval, yval - ci_width, yval + ci_width,
                            color=c, alpha=0.25)
        ax.set_xlabel("Time", size=14)
        ax.set_ylabel("Local Average Factor Coefficient", size=14)
        ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.0, 1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.figure.tight_layout()
        return ax

    def plot_residuals(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        apex_time = self._summary_statistics['apex_time_array']
        residuals = self._summary_statistics['residuals_array']
        ax.scatter(apex_time, residuals, s=15, edgecolors='black', alpha=0.5, color='teal')
        ax.set_xlabel("Time", size=14)
        ax.set_ylabel("Residuals", size=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        max_val = np.nanmax(residuals)
        min_val = np.nanmin(residuals)
        for t, mod in self.models.items():
            ax.vlines([mod.start, mod.end], min_val, max_val, colors='gray', lw=0.5, linestyles='--')

        ax.set_ylim(min_val - 1, max_val + 1)
        ax.hlines([-self.width_range.upper, self.width_range.upper],
                  apex_time.min(), apex_time.max(), linestyles='--', lw=0.5, color='gray')

        if self.interval_padding:
            ax.hlines([-self.width_range.upper - self.interval_padding, self.width_range.upper + self.interval_padding],
                      apex_time.min(), apex_time.max(), linestyles='dotted', lw=0.5, color='black')
        ax.figure.tight_layout()
        return ax

    def estimate_mean_error_from_interval(self) -> float:
        cis = self._summary_statistics['confidence_intervals']
        ys = self._summary_statistics['apex_time_array']
        # The widths of each interval
        widths = (cis[:, 1] - cis[:, 0]) / 2
        centroid_error = np.abs(cis - ys[:, None])
        error_from_nearest_interval_edge = np.clip(centroid_error - widths[:, None], 0, np.inf).min(axis=1)
        mean_error = np.nanmean(error_from_nearest_interval_edge)
        return mean_error

    def _reconstitute_proxies(self) -> List[GlycopeptideChromatogramProxy]:
        proxies = []

        summary_values = self._summary_statistics

        for i, (label, mass_shifts) in enumerate(summary_values['labels']):
            gp = FragmentCachingGlycopeptide(label)
            gc = gp.glycan_composition
            apex_time = summary_values['apex_time_array'][i]
            proxies.append(
                GlycopeptideChromatogramProxy(
                    gp.total_mass, apex_time, 1.0, gc, mass_shifts=mass_shifts, structure=gp
                )
            )
        return proxies


def unmodified_modified_predicate(x):
    return Unmodified in x.mass_shifts and Ammonium in x.mass_shifts


class EnsembleBuilder(TaskBase):
    aggregator: GlycoformAggregator
    points: Optional[np.ndarray]
    centers: OrderedDict
    models: OrderedDict

    def __init__(self, aggregator: GlycoformAggregator):
        self.aggregator = aggregator
        self.points = None
        self.centers = OrderedDict()
        self.models = OrderedDict()

    def estimate_width(self) -> float:
        try:
            width = int(max(np.nanmedian(v) for v in
                            self.aggregator.deltas().values()
                            if len(v) > 0)) + 1
        except ValueError:
            width = 2.0
        width *= 2.0
        self.log("... Estimated Region Width: %0.2f" % (width, ))
        return width

    def generate_points(self, width: float) -> np.ndarray:
        points = np.arange(
            int(self.aggregator.start_time) - 1,
            int(self.aggregator.end_time) + 1, width / 4.)
        if len(points) > 0 and points[-1] <= self.aggregator.end_time:
            points = np.append(points, self.aggregator.end_time + 1)
        return points

    def partition(self, width: Optional[float]=None, predicate: Optional[Callable]=None):
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
            if width_value is None:
                raise ValueError("width_value was None")
            self.points = self.generate_points(width_value)
        centers = OrderedDict()
        for point in self.points:
            w = width[point]
            centers[point] = GlycoformAggregator(
                list(filter(predicate,
                            self.aggregator.overlaps(point - w, point + w))))
        self.centers = centers
        return self

    def estimate_delta_widths(self, baseline: Optional[float]=None) -> OrderedDict:
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

    def _fit(self, i: int, point: float, members: List[GlycopeptideChromatogramProxy],
             point_members: List[List[GlycopeptideChromatogramProxy]],
             predicate: Callable[[Any], bool], model_type: Callable[..., ElutionTimeFitter],
             regularize: bool = False, resample: bool = False) -> Tuple[float, ElutionTimeFitter]:
        n = len(point_members)
        obs = list(filter(predicate, members))
        if not obs:
            return point, None

        m = model_type(
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
                m = model_type(obs, self.aggregator.factors, regularize=regularize)
                offset += 1
            self.debug(
                "... Borrowed %d observations from regions %d steps away for region at %0.3f (%d members, %r)" % (
                len(extra - set(members)), offset, point, len(members), m.data.shape))

        m = m.fit(resample=resample)
        return point, m

    def fit(self, predicate: Optional[Callable[[Any], bool]]=None,
            model_type: Optional[Callable[..., ElutionTimeFitter]]=None,
            regularize: bool=False, resample: bool=False, n_workers: Optional[int]=None):
        if model_type is None:
            model_type = AbundanceWeightedPeptideFactorElutionTimeFitter
        if predicate is None:
            predicate = bool
        if n_workers is None:
            n_workers = min(cpu_count(), 6)

        models: OrderedDict[float, ElutionTimeFitter] = OrderedDict()
        point_members = list(self.centers.items())

        tasks: OrderedDict[float, futures.Future[Tuple[float,
                                                       ElutionTimeFitter]]] = OrderedDict()
        with futures.ThreadPoolExecutor(n_workers, thread_name_prefix="ElutionTimeModelFitter") as executor:
            for i, (point, members) in enumerate(point_members):
                m: ElutionTimeFitter

                result = executor.submit(
                    self._fit, i, point, members, point_members, predicate,
                    model_type, regularize, resample)
                tasks[point] = result

            for _, result in tasks.items():
                point, m = result.result()
                if m is None:
                    continue
                models[point] = m
        self.models = models
        return self

    def merge(self):
        return ModelEnsemble(self.models)


DeltaParam = namedtuple(
    "DeltaParam", ("mean", "sdev", "size", "values", "weights"))


class LocalShiftGraph(object):
    chromatograms: List
    max_distance: float
    key_cache: GlycanCompositionDeltaCache
    distance_cache: DistanceCache
    edges: DefaultDict[frozenset, List]
    shift_to_delta: DefaultDict[str, List[Tuple[GlycopeptideChromatogramProxy, frozenset]]]
    groups: GlycoformAggregator
    key_cache: Dict
    delta_params: Dict[str, DeltaParam]

    def __init__(self, chromatograms, max_distance=3, key_cache=None, distance_cache=None, delta_cache=None):
        if distance_cache is None:
            distance_cache = DistanceCache(composition_distance)
        if delta_cache is None:
            delta_cache = GlycanCompositionDeltaCache()
        self.chromatograms = chromatograms
        self.max_distance = max_distance
        self.edges = defaultdict(list)
        self.shift_to_delta = defaultdict(list)
        self.groups = GlycoformAggregator(chromatograms)
        self.key_cache = key_cache or {}
        self.distance_cache = distance_cache
        self.delta_cache = delta_cache

        self.build_edges()
        self.delta_params = self.estimate_delta_distribution()

    def build_edges(self):
        for key, group in self.groups.by_peptide.items():
            base_time = 0.0
            a: GlycopeptideChromatogramProxy
            b: GlycopeptideChromatogramProxy
            for a, b in itertools.permutations(group, 2):
                distance = self.distance_cache(
                    a.glycan_composition, b.glycan_composition)[0]
                if distance > self.max_distance or distance == 0:
                    continue
                delta_comp = self.delta_cache(a.glycan_composition, b.glycan_composition)
                # Construct the structure property directly without paying property overhead
                structure_key = (key + delta_comp.serialize())
                if structure_key in self.key_cache:
                    structure = self.key_cache[structure_key]
                else:
                    structure = FragmentCachingGlycopeptide(structure_key)
                    self.key_cache[structure_key] = structure
                rec = GlycopeptideChromatogramProxy(
                    a.weighted_neutral_mass, base_time + a.apex_time - b.apex_time,
                    np.sqrt(a.total_signal * b.total_signal),
                    delta_comp,
                    structure=structure,
                    peptide_key=key,
                    weight=float(distance) ** 4 + (a.weight + b.weight)
                )
                rec._structure = structure

                a_tag = a.tag
                b_tag = b.tag
                if a_tag is not None and b_tag is not None:
                    edge_key = frozenset((a_tag, b_tag))
                else:
                    edge_key = frozenset((str(a.structure), str(b.structure)))
                delta_key = delta_comp
                self.edges[edge_key].append(rec)
                # Use string form to discard 0-value keys
                self.shift_to_delta[str(delta_key)].append((rec, edge_key))

    def estimate_delta_distribution(self) -> Dict[str, DeltaParam]:
        params = {}
        for shift, delta_chroms in self.shift_to_delta.items():
            apex_times = array.array('f')
            weights = array.array('f')
            for rec, _key in delta_chroms:
                apex_times.append(rec.apex_time)
                weights.append(rec.total_signal if rec.total_signal > 1 else (1 + 1e-13))

            apex_times = np.array(apex_times)
            weights = np.log10(weights)

            W = weights.sum()
            weighted_mean: float = np.dot(apex_times, weights) / W
            weighted_sdev: float = np.sqrt(
                weights.dot((apex_times - weighted_mean) ** 2) / (W - 1)
            )
            if weighted_sdev > abs(weighted_mean):
                tmp = weighted_sdev
                weighted_sdev = max(abs(weighted_mean), 1.0)
                logger.debug(
                    "Delta parameter %r's standard deviation was over-dispersed %r/%r. Capping at %r",
                    shift, tmp, weighted_mean, weighted_sdev
                )

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
                W = weights.sum()
                if len(weights):
                    weighted_mean = np.dot(apex_times, weights) / W
                    weighted_sdev = np.sqrt(weights.dot((apex_times - weighted_mean) ** 2) / (W - 1))
                else:
                    weighted_mean = 0
                    weighted_sdev = 1

            params[shift] = DeltaParam(weighted_mean, weighted_sdev, len(apex_times), apex_times, weights)
        return params

    def is_composite(self, shift) -> bool:
        value = HashableGlycanComposition.parse(shift)
        return sum(map(abs, value.values())) > 1

    def composite_to_individuals(self, shift) -> List[Tuple[HashableGlycanComposition, int]]:
        shift = HashableGlycanComposition.parse(shift)
        return [(HashableGlycanComposition({k: v / abs(v)}), v) for k, v in shift.items()]

    def compose_shift(self, shift) -> Tuple[float, float]:
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

    def process_edges(self) -> set:
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
                except KeyError:
                    # We don't have a component edge to support this composite edge, so we don't
                    # have enough information in the training data, so we're unable QC these edges.
                    # This is different from when we have a singular observation of any component,
                    # which may be a logical failure here.
                    # logger.debug("Missing component key %r", err)
                    pass

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
    def __init__(self, chromatograms, factors=None, scale=1, transform=None, width_range=None,
                 regularize=False, max_distance=3, key_cache=None, distance_cache=None,
                 delta_cache=None):
        if key_cache is None:
            key_cache = {}
        if distance_cache is None:
            distance_cache = DistanceCache(composition_distance)
        if delta_cache is None:
            delta_cache = GlycanCompositionDeltaCache()
        self.key_cache = key_cache
        self.distance_cache = distance_cache
        self.delta_cache = delta_cache
        self.max_distance = max_distance
        if chromatograms:
            recs, groups = self.build_deltas(chromatograms, max_distance=max_distance)
        else:
            recs = []
            groups = GlycoformAggregator()
        self.groups = groups
        self.peptide_offsets = {}
        super(RelativeShiftFactorElutionTimeFitter, self).__init__(
            recs, factors, scale=scale, transform=transform,
            width_range=width_range, regularize=regularize)

    def drop_chromatograms(self):
        super(RelativeShiftFactorElutionTimeFitter, self).drop_chromatograms()
        self.groups = None
        return self

    def __getstate__(self):
        state = super(RelativeShiftFactorElutionTimeFitter, self).__getstate__()
        state['groups'] = self.groups
        state['peptide_offsets'] = self.peptide_offsets
        state['max_distance'] = self.max_distance
        return state

    def __setstate__(self, state):
        super(RelativeShiftFactorElutionTimeFitter, self).__setstate__(state)
        self.groups = state['groups']
        self.peptide_offsets = state['peptide_offsets']
        self.max_distance = state['max_distance']
        self.key_cache = dict()

    def build_deltas(self, chromatograms, max_distance):
        recs = []
        groups = GlycoformAggregator(chromatograms)
        for key, group in groups.by_peptide.items():
            base_time = 0.0
            for a, b in itertools.permutations(group, 2):
                distance = self.distance_cache(
                    a.glycan_composition, b.glycan_composition)[0]
                if distance > max_distance or distance == 0:
                    continue
                delta_comp = self.delta_cache(a.glycan_composition, b.glycan_composition)
                structure = (key + str(delta_comp))
                if structure in self.key_cache:
                    structure = self.key_cache[structure]
                else:
                    structure_key = structure
                    structure = FragmentCachingGlycopeptide(structure)
                    self.key_cache[structure_key] = structure
                rec = GlycopeptideChromatogramProxy(
                    a.weighted_neutral_mass, base_time + a.apex_time - b.apex_time,
                    np.sqrt(a.total_signal * b.total_signal),
                    delta_comp, structure=structure,
                    weight=float(distance) ** 4 + (a.weight + b.weight),
                    peptide_key=a.peptide_key
                )
                recs.append(rec)
        if len(recs) == 0:
            raise ValueError("No deltas constructed")
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
        if len(self.groups) == 0:
            self.start = 0.0
            self.end = 0.0
            self.centroid = 0.0
        else:
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
        graph = LocalShiftGraph(
            chromatograms, max_distance,
            key_cache=self.key_cache,
            distance_cache=self.distance_cache)
        graph.process_edges()
        groups = graph.groups
        recs = graph.select_edges()
        if len(recs) == 0:
            raise ValueError("No deltas constructed")
        return recs, groups


def mask_in(full_set: List[GlycopeptideChromatogramProxy], incoming: List[GlycopeptideChromatogramProxy]) -> List[GlycopeptideChromatogramProxy]:
    """
    Replace the entries in ``full_set`` with the entries in ``incoming`` if their tags match,
    otherwise use the original value.

    Returns
    -------
    List[GlycopeptideChromatogramProxy]
    """
    incoming_map = {i.tag: i for i in incoming}
    out = []
    for chrom in full_set:
        if chrom.tag in incoming_map:
            out.append(incoming_map[chrom.tag])
        else:
            out.append(chrom)
    return out


# The self paramter is included as this function is later bound to a class directly
# so it gets made into a method
def model_type_dispatch(self, chromatogams: List[GlycopeptideChromatogramProxy],
                        factors: Optional[List[str]]=None,
                        scale: float=1,
                        transform=None,
                        width_range: Optional[IntervalRange]=None,
                        regularize: bool=False,
                        key_cache=None,
                        distance_cache: Optional[DistanceCache]=None,
                        delta_cache: Optional[GlycanCompositionDeltaCache]=None):
    try:
        model = RelativeShiftFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize,
            max_distance=3, key_cache=key_cache, distance_cache=distance_cache,
            delta_cache=delta_cache)
        if model.data.shape[0] == 0:
            raise ValueError("No relative data points")
        return model
    except (ValueError, AssertionError):
        model = AbundanceWeightedPeptideFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize)
        return model


# The self paramter is included as this function is later bound to a class directly
# so it gets made into a method
def model_type_dispatch_outlier_remove(self, chromatogams: List[GlycopeptideChromatogramProxy],
                                       factors: Optional[List[str]]=None,
                                       scale: float=1,
                                       transform=None,
                                       width_range: Optional[IntervalRange]=None,
                                       regularize: bool=False,
                                       key_cache=None,
                                       distance_cache: Optional[DistanceCache]=None,
                                       delta_cache: Optional[GlycanCompositionDeltaCache]=None):
    try:
        model = LocalOutlierFilteringRelativeShiftFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize,
            max_distance=3, key_cache=key_cache, distance_cache=distance_cache,
            delta_cache=delta_cache)
        if model.data.shape[0] == 0:
            raise ValueError("No relative data points")
        return model
    except (ValueError, AssertionError):
        model = AbundanceWeightedPeptideFactorElutionTimeFitter(
            chromatogams, factors, scale, transform, width_range, regularize)
        return model


class GlycopeptideElutionTimeModelBuildingPipeline(TaskBase):
    '''A local regression-based ensemble of overlapping
    retention time models.
    '''

    model_type = model_type_dispatch_outlier_remove

    aggregate: GlycoformAggregator
    calibration_alpha: float
    _current_model: ModelEnsemble
    model_history: List[ModelEnsemble]
    n_workers: int

    revised_tags: Set[int]
    revision_history: List[List[GlycopeptideChromatogramProxy]]
    revision_rules: RevisionRuleList
    revision_validator: RevisionValidatorBase

    valid_glycans: Optional[Set[HashableGlycanComposition]]
    initial_filter: Callable[..., bool]

    key_cache: Dict
    distance_cache: DistanceCache
    delta_cache: GlycanCompositionDeltaCache

    def __init__(self, aggregate, calibration_alpha=0.001, valid_glycans=None,
                 initial_filter=unmodified_modified_predicate, revision_validator=None,
                 n_workers=None):
        if n_workers is None:
            n_workers = cpu_count()
        if revision_validator is None:
            revision_validator = PeptideYUtilizationPreservingRevisionValidator()
        self.aggregate = aggregate
        self.calibration_alpha = calibration_alpha
        self.n_workers = n_workers
        self._current_model = None
        self.model_history = []
        self.revision_history = []
        self.revised_tags = set()
        # The indexed variant doesn't work well without expanding the space to also
        # include revision rules and requires that the chromatogram proxies need to
        # have special copy logic
        self.valid_glycans = ValidatedGlycome(valid_glycans) if valid_glycans else None
        self.initial_filter = initial_filter
        self.key_cache = {}
        self.revision_validator = revision_validator

        self.distance_cache = DistanceCache(composition_distance)
        self.delta_cache = GlycanCompositionDeltaCache()
        self.revision_rules = self._default_rules()
        self._check_rules()

    def _default_rules(self):
        return RevisionRuleList([
            AmmoniumMaskedRule,
            AmmoniumUnmaskedRule,
            AmmoniumMaskedNeuGcRule,
            AmmoniumUnmaskedNeuGcRule,
            IsotopeRule,
            IsotopeRule2,
            IsotopeRuleNeuGc,
            HexNAc2NeuAc2ToHex6AmmoniumRule,
            HexNAc2NeuAc2ToHex6Deoxy,
            HexNAc2Fuc1NeuAc2ToHex7,
            SulfateToPhosphateRule,
            PhosphateToSulfateRule,
            Sulfate1HexNAc2ToHex3Rule,
            Hex3ToSulfate1HexNAc2Rule,
            Phosphate1HexNAc2ToHex3Rule,
            Hex3ToPhosphate1HexNAc2Rule,
        ]).with_cache()

    def _check_rules(self):
        if self.valid_glycans:
            fucose = 0
            dhex = 0
            for gc in self.valid_glycans:
                fucose += gc['Fuc']
                dhex += gc['dHex']
            if fucose == 0 and dhex > 0:
                self.revision_rules = self.revision_rules.modify_rules({libreviser.fuc: libreviser.dhex})
            elif dhex == 0 and fucose > 0:
                # no-op, we don't need to make changes to the current
                # rules.
                pass
            else:
                self.log("No Fuc or d-Hex detected in valid glycans, no rule modifications inferred")
            n_rules_before = len(self.revision_rules)
            self.revision_rules = self.revision_rules.filter_defined(self.valid_glycans)
            n_rules_after = len(self.revision_rules)
            self.log(f"Using {n_rules_after} revision rules ({n_rules_before - n_rules_after} removed).")

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, value):
        if self._current_model is not None and value is not self._current_model:
            pass
            # self.model_history.append(self._current_model)
        self._current_model = value

    def fit_first_model(self, regularize=False):
        model = self.fit_model(
            self.aggregate,
            self.initial_filter,
            regularize=regularize)
        return model

    def fit_model(self, aggregate, predicate, regularize=False, resample=False):
        builder = EnsembleBuilder(aggregate)
        w = builder.estimate_width()
        # If width changing over time is allowed, approximate narrow windows will inevitably produce
        # skewed results. This is also true of the first and last set of windows, regardless, especially
        # when sparse.
        builder.partition(w)
        model_type = partial(
            self.model_type,
            key_cache=self.key_cache,
            distance_cache=self.distance_cache,
            delta_cache=self.delta_cache)
        builder.fit(predicate, model_type=model_type,
                    regularize=regularize, resample=resample,
                    n_workers=self.n_workers)
        model = builder.merge()
        model.calibrate_prediction_interval(alpha=self.calibration_alpha)
        self.log("... Calibrated Prediction Intervals: %0.3f-%0.3f" % (
            model.width_range.lower, model.width_range.upper))
        return model

    def fit_model_with_revision(self, source, revision, regularize=False):
        masked_in_all_recs = mask_in(source, revision)
        aggregate = GlycoformAggregator(masked_in_all_recs)
        tags = {t.tag for t in revision}
        return self.fit_model(aggregate, lambda x: x.tag in tags, regularize=regularize)

    def make_reviser(self, model, chromatograms):
        reviser = IntervalModelReviser(
            model, self.revision_rules,
            chromatograms,
            valid_glycans=self.valid_glycans,
            revision_validator=self.revision_validator)
        return reviser

    def revise_with(self, model, chromatograms, delta=0.35, min_time_difference=None, verbose=True):
        if min_time_difference is None:
            min_time_difference = max(model.width_range.lower, 0.5)

        reviser = self.make_reviser(model, chromatograms)
        reviser.evaluate()
        revised = reviser.revise(0.2, delta,  min_time_difference)
        if verbose:
            self.debug("... Revising Observations")
            k = 0
            local_aggregate = GlycoformAggregator(chromatograms)
            for new, old in zip(revised, chromatograms):
                if new.structure != old.structure:
                    self.revised_tags.add(new.tag)
                    neighbor_count = len(local_aggregate[old])
                    k += 1
                    self.log(
                        "...... %s: %0.2f %s (%0.2f) => %s (%0.2f) %0.2f/%0.2f with %d references" % (
                            new.tag, new.apex_time,
                            old.structure, model.predict(old),
                            new.glycan_composition, model.predict(new),
                            model.score_interval(old, alpha=0.01),
                            model.score_interval(new, alpha=0.01),
                            neighbor_count - 1,
                        )
                    )
            self.debug("... Updated %d assignments" % (k, ))
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
        return np.array(list(chromatograms))[
            np.where((coverages >= threshold) | nans | np.isclose(coverages, threshold))[0]
        ]

    def reweight(self, model, chromatograms, base_weight=0.2):
        reweighted = []
        total = 0.0
        total_ignored = 0
        for rec in chromatograms:
            rec = rec.copy()
            s = w = model.score_interval(rec, 0.01)
            if np.isnan(w):
                s = w = 0
            total_ignored += w == 0
            rec.weight = w + 1e-6
            if np.isnan(rec.weight):
                rec.weight = base_weight
            reweighted.append(rec)
            total += s
        self.log("... Total Score for %d chromatograms = %f (%d ignored)" %
                 (len(chromatograms), total, total_ignored))
        return reweighted

    def make_groups_from(self, chromatograms):
        return GlycoformAggregator(chromatograms).by_peptide.values()

    def find_uncovered_group_members(self, chromatograms, coverages, min_size=2, seen=None):
        recs = []
        for chrom, cov in zip(chromatograms, coverages):
            if seen and chrom.tag in seen:
                continue
            if not np.isclose(cov, 1.0) and len(self.aggregate[chrom]) > min_size:
                recs.append(chrom)
        return recs

    def validate_revisions(self, model, revisions, delta=0.35, min_time_difference=None):
        originals = []
        for index, revision in revisions:
            _rule, original = revision.revised_from
            originals.append(original)
        self.log("... Validating revisions with final model")
        recalculated = self.revise_with(
            model, originals, delta=delta, min_time_difference=min_time_difference, verbose=False)

        result = []
        for i, checked in enumerate(recalculated):
            index, revision = revisions[i]
            _rule, original = revision.revised_from

            checked_rt = model.predict(checked)
            revision_rt = model.predict(revision)
            checked_score = model.score_interval(checked, alpha=0.01)
            revision_score = model.score_interval(revision, alpha=0.01)

            if abs(checked_rt - revision_rt) < min_time_difference:
                continue
            if abs(checked_score - revision_score) < delta:
                continue

            if checked_score == revision_score == 0:
                if abs(revision_rt - revision.apex_time) / abs(checked_rt - revision.apex_time) < 0.5:
                    continue

            if revision.glycan_composition != checked.glycan_composition:
                self.log("...... Rejected Revision: %s" % (revision.structure, ))
                self.log(
                   "...... %s: %0.2f %s (%0.2f) => %s (%0.2f) %0.2f/%0.2f" % (
                       revision.tag, revision.apex_time,
                       revision.glycan_composition, model.predict(revision),
                       checked.glycan_composition, model.predict(checked),
                       model.score_interval(revision, alpha=0.01),
                       model.score_interval(checked, alpha=0.01),
                   )
               )
            if checked.glycan_composition == original.glycan_composition:
                checked = original

            result.append((index, checked))
        return result

    def estimate_fdr(self, chromatograms, model, rules=None):
        if rules is None:
            rules = self.revision_rules
        estimators = []
        for rule in rules:
            self.log("... Fitting decoys for %r" % (rule, ))
            estimator = RuleBasedFDREstimator(
                    rule, chromatograms, model, self.valid_glycans)
            estimators.append(estimator)

        fitted_rules = RevisionRuleList(estimators)
        self.log("... Re-calibrating score from decoy residuals")
        try:
            residual_fdr = ResidualFDREstimator(fitted_rules, model)
            width = np.abs(residual_fdr.bounds_for_probability(0.95)).max()
            if model.width_range:
                delta = width - model.width_range.upper
                max_delta = min(model.model_width / 2, delta)
                if max_delta != delta:
                    self.log("... Maximum padding interval exceeded (%0.3f > %0.3f)" % (
                            delta, max_delta))
                    delta = max_delta
                if delta > 0:
                    model.interval_padding = delta
                    self.log("... Setting padding interval from residual FDR: %0.3f" %
                            (delta, ))
            for estimator in fitted_rules:
                estimator.prepare()
                t05 = estimator.score_for_fdr(0.05)
                t01 = estimator.score_for_fdr(0.01)
                self.log("... FDR for {}".format(estimator.rule))
                self.log("...... 5%: {:0.2f}    1%: {:0.2f}".format(t05, t01))
                self.log("... Fitting relationship over time")
                estimator.fit_over_time()
        except ValueError as err:
            logger.error("Unable to fit Residual FDR for RT Model: %s", err, exc_info=False)
            residual_fdr = None
        model.residual_fdr = residual_fdr
        return fitted_rules

    def extract_rules_used(self, chromatograms):
        rules_used = set()
        for record in chromatograms:
            if record.revised_from:
                rule, _ = record.revised_from
                rules_used.add(rule)
        return rules_used

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
        coverages = self.compute_model_coverage(
            self.current_model, all_records)

        # Select only those chromatograms that have 100% coverage (always defined in the model)
        # but regardless of modification state.
        covered_recs = self.subset_aggregate_by_coverage(
            all_records, coverages, 1.0)

        # Attempt to revise chromatograms which are covered by the current model
        revised_recs = self.revise_with(
            self.current_model, covered_recs, 0.35, max(self.current_model.width_range.lower * 0.5, 0.5))

        # Update the list of chromatogram records
        all_records = mask_in(list(self.aggregate), revised_recs)

        # Fit a new model on the revised data
        self.current_model = model = self.fit_model_with_revision(
            all_records, revised_recs, regularize=regularize)

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

            if len(covered_recs) == len(last_covered):
                self.log("No new observations added, skipping")
                continue

            new = {c.tag for c in covered_recs} - \
                {c.tag for c in last_covered}
            last_covered = covered_recs
            self.log("... Covering %d chromatograms at threshold %0.2f" % (
                len(covered_recs), coverage_threshold))
            self.debug("... Added %d new tags" % len(new))
            revised_recs = self.revise_with(
                model, covered_recs, revision_threshold,
                max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))
            revised_recs = self.reweight(model, revised_recs, base_weight)
            all_records = mask_in(all_records, revised_recs)

            self.current_model = model = self.fit_model_with_revision(
                all_records, revised_recs, regularize=regularize)

        if final_round:
            self.log("Open Update Round")
            coverages = self.compute_model_coverage(model, all_records)
            coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
            covered_recs = self.subset_aggregate_by_coverage(
                all_records, coverages, coverage_threshold)

            extra_recs = self.find_uncovered_group_members(
                all_records,
                coverages
            )

            self.debug(f"...... Found {len(covered_recs)} covered chromatograms")
            self.debug(f"...... Found {len(extra_recs)} additional chromatograms")

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
                    new = {c.tag for c in covered_recs} - \
                        {c.tag for c in last_covered}
                    last_covered = covered_recs
                self.log("... Covering %d chromatograms at threshold %0.2f" % (
                    len(covered_recs), coverage_threshold))
                self.log("... Added %d new tags" % len(new))
                revised_recs = self.revise_with(
                    model, covered_recs, revision_threshold,
                    max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))
                revised_recs = self.reweight(model, revised_recs, base_weight)
                all_records = mask_in(all_records, revised_recs)

                self.current_model = model = self.fit_model_with_revision(
                    all_records, revised_recs, regularize=regularize)

            self.log("Final Update Round")
            coverages = self.compute_model_coverage(model, all_records)
            coverage_threshold = (1.0 - (i * 1.0 / (k_scale * k)))
            covered_recs = self.subset_aggregate_by_coverage(
                all_records, coverages, coverage_threshold)

            extra_recs = self.find_uncovered_group_members(
                all_records, coverages)

            self.log("... Added %d new tags" % len(extra_recs))
            covered_recs = np.concatenate((covered_recs, extra_recs))
            covered_recs = self.reweight(model, covered_recs, base_weight=0.01)

            all_records = mask_in(all_records, covered_recs)
            self.current_model = model = self.fit_model_with_revision(
                all_records, covered_recs, regularize=regularize)
            revised_recs = self.revise_with(
                model, covered_recs, revision_threshold,
                max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))

            all_records = mask_in(all_records, revised_recs)

        self.log("Estimating summary statistics")
        model._compute_summary_statistics()

        try:
            mean_error_from_interval = model.estimate_mean_error_from_interval()
        except IndexError:
            self.log("Unable to fit final model")
            return None, all_records

        self.log("... Adding padding mean interval error: %0.3f" % (mean_error_from_interval, ))
        model.interval_padding = mean_error_from_interval
        self.log("Last revision")
        revised_recs = self.revise_with(
            model, revised_recs, revision_threshold,
            max(self.current_model.width_range.lower * delta_time_scale, minimum_delta))
        all_records = mask_in(all_records, revised_recs)

        was_revised = [(i, record) for i, record in enumerate(all_records) if record.revised_from]
        for i, record in self.validate_revisions(model, was_revised, revision_threshold,
                                                 max(self.current_model.width_range.lower * delta_time_scale,
                                                     minimum_delta)):
            all_records[i] = record
        rules_used = self.extract_rules_used(all_records)
        self.log("Estimating FDR for revision rules")
        model._summary_statistics['fdr_estimates'] = self.estimate_fdr(all_records, model, rules_used)
        self.log("Final Model Variance Explained: %0.3f" % (model.R2(), ))
        return model, all_records


ModelBuildingPipeline = GlycopeptideElutionTimeModelBuildingPipeline


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
