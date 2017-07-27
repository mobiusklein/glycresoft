from collections import OrderedDict
from itertools import product

import six

import numpy as np

from scipy.optimize import leastsq
from numpy import pi, sqrt, exp
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

from ms_peak_picker import search

from .base import ScoringFeatureBase, epsilon


MIN_POINTS = 5
MAX_POINTS = 2000
SIGMA_EPSILON = 1e-3


def prepare_arrays_for_linear_fit(x, y):
    X = np.vstack((np.ones(len(x)), np.array(x))).T
    Y = np.array(y)
    return X, Y


def linear_regression_fit(x, y, prepare=False):
    if prepare:
        x, y = prepare_arrays_for_linear_fit(x, y)
    B = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    Yhat = x.dot(B)
    return Yhat


def linear_regression_residuals(x, y):
    X, Y = prepare_arrays_for_linear_fit(x, y)
    Yhat = linear_regression_fit(X, Y)
    return (Y - Yhat) ** 2


def flat_line_residuals(y):
    residuals = (
        (y - ((y.max() + y.min()) / 2.))) ** 2
    return residuals


class PeakShapeModelBase(object):
    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)

    @classmethod
    def nargs(self):
        return six.get_function_code(self.shape).co_argcount - 1

    @classmethod
    def get_min_points(self):
        return getattr(self, "min_points", self.nargs() + 1)


def gaussian_shape(xs, center, amplitude, sigma):
    if sigma == 0:
        sigma = SIGMA_EPSILON
    norm = (amplitude) / (sigma * sqrt(2 * pi)) * \
        exp(-((xs - center) ** 2) / (2 * sigma ** 2))
    return norm


class GaussianModel(PeakShapeModelBase):
    min_points = 6

    @staticmethod
    def fit(params, xs, ys):
        center, amplitude, sigma = params
        return ys - GaussianModel.shape(xs, center, amplitude, sigma)

    @staticmethod
    def shape(xs, center, amplitude, sigma):
        return gaussian_shape(xs, center, amplitude, sigma)

    @staticmethod
    def guess(xs, ys):
        center = np.average(xs, weights=ys / ys.sum())
        height_at = np.abs(xs - center).argmin()
        apex = ys[height_at]
        sigma = np.abs(center - xs[[search.nearest_left(ys, apex / 2, height_at),
                                    search.nearest_right(ys, apex / 2, height_at + 1)]]).sum()
        return center, apex, sigma

    @staticmethod
    def params_to_dict(params):
        center, amplitude, sigma = params
        return OrderedDict((("center", center), ("amplitude", amplitude), ("sigma", sigma)))

    @staticmethod
    def center(params_dict):
        return params_dict['center']

    @staticmethod
    def spread(params_dict):
        return params_dict['sigma']


def skewed_gaussian_shape(xs, center, amplitude, sigma, gamma):
    if sigma == 0:
        sigma = SIGMA_EPSILON
    norm = (amplitude) / (sigma * sqrt(2 * pi)) * \
        exp(-((xs - center) ** 2) / (2 * sigma ** 2))
    skew = (1 + erf((gamma * (xs - center)) / (sigma * sqrt(2))))
    return norm * skew


class SkewedGaussianModel(PeakShapeModelBase):
    @staticmethod
    def fit(params, xs, ys):
        center, amplitude, sigma, gamma = params
        return ys - SkewedGaussianModel.shape(xs, center, amplitude, sigma, gamma) * (
            sigma / 2. if abs(sigma) > 2 else 1.)

    @staticmethod
    def guess(xs, ys):
        center = np.average(xs, weights=ys / ys.sum())
        height_at = np.abs(xs - center).argmin()
        apex = ys[height_at]
        sigma = np.abs(center - xs[[search.nearest_left(ys, apex / 2, height_at),
                                    search.nearest_right(ys, apex / 2, height_at + 1)]]).sum()
        gamma = 1
        return center, apex, sigma, gamma

    @staticmethod
    def params_to_dict(params):
        center, amplitude, sigma, gamma = params
        return OrderedDict((("center", center), ("amplitude", amplitude), ("sigma", sigma), ("gamma", gamma)))

    @staticmethod
    def shape(xs, center, amplitude, sigma, gamma):
        return skewed_gaussian_shape(xs, center, amplitude, sigma, gamma)

    @staticmethod
    def center(params_dict):
        return params_dict['center']

    @staticmethod
    def spread(params_dict):
        return params_dict['sigma']


class PenalizedSkewedGaussianModel(SkewedGaussianModel):
    @staticmethod
    def fit(params, xs, ys):
        center, amplitude, sigma, gamma = params
        return ys - PenalizedSkewedGaussianModel.shape(xs, center, amplitude, sigma, gamma) * (
            sigma / 2. if abs(sigma) > 2 else 1.) * (gamma / 2. if abs(gamma) > 40 else 1.) * (
            center if center > xs[-1] or center < xs[0] else 1.)


def bigaussian_shape(xs, center, amplitude, sigma_left, sigma_right):
    if sigma_left == 0:
        sigma_left = SIGMA_EPSILON
    if sigma_right == 0:
        sigma_right = SIGMA_EPSILON
    ys = np.zeros_like(xs, dtype=np.float32)
    left_mask = xs < center
    ys[left_mask] = amplitude * np.exp(-(xs[left_mask] - center) ** 2 / (2 * sigma_left ** 2)) * sqrt(2 * pi)
    right_mask = xs > center
    ys[right_mask] = amplitude * np.exp(-(xs[right_mask] - center) ** 2 / (2 * sigma_right ** 2)) * sqrt(2 * pi)
    return ys


class BiGaussianModel(PeakShapeModelBase):

    @staticmethod
    def center(params_dict):
        return params_dict['center']

    @staticmethod
    def spread(params_dict):
        return (params_dict['sigma_left'] + params_dict['sigma_right']) / 2.

    @staticmethod
    def shape(xs, center, amplitude, sigma_left, sigma_right):
        return bigaussian_shape(xs, center, amplitude, sigma_left, sigma_right)

    @staticmethod
    def fit(params, xs, ys):
        center, amplitude, sigma_left, sigma_right = params
        return ys - BiGaussianModel.shape(
            xs, center, amplitude, sigma_left, sigma_right) * (
            center if center > xs[-1] or center < xs[0] else 1.)

    @staticmethod
    def params_to_dict(params):
        center, amplitude, sigma_left, sigma_right = params
        return OrderedDict(
            (("center", center), ("amplitude", amplitude), ("sigma_left", sigma_left), ("sigma_right", sigma_right)))

    @staticmethod
    def guess(xs, ys):
        center = np.average(xs, weights=ys / ys.sum())
        height_at = np.abs(xs - center).argmin()
        apex = ys[height_at]
        sigma = np.abs(center - xs[[search.nearest_left(ys, apex / 2, height_at),
                                    search.nearest_right(ys, apex / 2, height_at + 1)]]).sum()
        return center, apex, sigma, sigma


class FittedPeakShape(object):
    def __init__(self, params, shape_model):
        self.params = params
        self.shape_model = shape_model

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def __iter__(self):
        return iter(self.params)

    def shape(self, xs):
        return self.shape_model.shape(xs, **self.params)

    def __getitem__(self, key):
        return self.params[key]

    def __repr__(self):
        return "Fitted{self.shape_model.__class__.__name__}({params})".format(
            self=self, params=", ".join("%s=%0.3f" % (k, v) for k, v in self.params.items()))

    @property
    def center(self):
        return self['center']

    @property
    def amplitude(self):
        return self['amplitude']


class ChromatogramShapeFitterBase(ScoringFeatureBase):
    feature_type = "line_score"

    def __init__(self, chromatogram, smooth=True, fitter=PenalizedSkewedGaussianModel()):
        self.chromatogram = chromatogram
        self.smooth = smooth
        self.xs = None
        self.ys = None
        self.line_test = None
        self.off_center = None
        self.shape_fitter = fitter

    def is_invalid(self):
        n = len(self.chromatogram)
        return n < MIN_POINTS or n < self.shape_fitter.get_min_points()

    def handle_invalid(self):
        self.line_test = 1 - 5e-6

    def extract_arrays(self):
        self.xs, self.ys = self.chromatogram.as_arrays()
        if self.smooth:
            self.ys = gaussian_filter1d(self.ys, 1)
        if len(self.xs) > MAX_POINTS:
            new_xs = np.linspace(self.xs.min(), self.xs.max(), MAX_POINTS)
            new_ys = np.interp(new_xs, self.xs, self.ys)
            self.xs = new_xs
            self.ys = new_ys
            self.ys = gaussian_filter1d(self.ys, 1)

    def compute_residuals(self):
        return NotImplemented

    def null_model_residuals(self):
        residuals = linear_regression_residuals(self.xs, self.ys)
        return residuals

    def perform_line_test(self):
        residuals = self.compute_residuals()
        line_test = (residuals ** 2).sum() / (
            (self.null_model_residuals()).sum())
        self.line_test = max(line_test, 1e-5)

    def plot(self, ax=None):
        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1)
        ob1 = ax.plot(self.xs, self.ys, label='Observed')[0]
        ob2 = ax.scatter(self.xs, self.ys, label='Observed')
        f1 = ax.plot(self.xs, self.compute_fitted(), label='Fitted')[0]
        r1 = ax.plot(self.xs, self.compute_residuals(), label='Residuals')[0]
        ax.legend(
            (
                (ob1, ob2),
                (f1,),
                (r1,)
            ), ("Observed", "Fitted", "Residuals")
        )
        return ax

    @property
    def fit_parameters(self):
        raise NotImplementedError()

    @classmethod
    def score(cls, chromatogram, *args, **kwargs):
        return max(1 - cls(chromatogram).line_test, epsilon)


class ChromatogramShapeFitter(ChromatogramShapeFitterBase):
    def __init__(self, chromatogram, smooth=True, fitter=PenalizedSkewedGaussianModel()):
        super(ChromatogramShapeFitter, self).__init__(chromatogram, smooth=smooth, fitter=fitter)

        self.params = None
        self.params_dict = None

        if self.is_invalid():
            self.handle_invalid()
        else:
            self.extract_arrays()
            self.peak_shape_fit()
            self.perform_line_test()
            self.off_center_factor()

    @property
    def fit_parameters(self):
        return self.params_dict

    def __repr__(self):
        return "ChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)

    def off_center_factor(self):
        center = self.shape_fitter.center(self.params_dict)
        spread = self.shape_fitter.spread(self.params_dict)
        self.off_center = abs(1 - abs(1 - (2 * abs(
            self.xs[self.ys.argmax()] - center) / abs(spread))))
        if self.off_center > 1:
            self.off_center = 1. / self.off_center
        self.line_test /= self.off_center

    def compute_residuals(self):
        return self.shape_fitter.fit(self.params, self.xs, self.ys)

    def compute_fitted(self):
        return self.shape_fitter.shape(self.xs, **self.params_dict)

    def peak_shape_fit(self):
        xs, ys = self.xs, self.ys
        params = self.shape_fitter.guess(xs, ys)
        fit = leastsq(self.shape_fitter.fit,
                      params, (xs, ys))
        params = fit[0]
        self.params = params
        self.params_dict = FittedPeakShape(self.shape_fitter.params_to_dict(params), self.shape_fitter)

    def iterfits(self):
        yield self.compute_fitted()


def shape_fit_test(chromatogram, smooth=True):
    return ChromatogramShapeFitter(chromatogram, smooth).line_test


def peak_indices(x, min_height=0):
    """Find the index of local maxima.

    Parameters
    ----------
    x : np.ndarray
        Data to find local maxima in
    min_height : float, optional
        Minimum peak height

    Returns
    -------
    np.ndarray[int]
        Indices of maxima in x

    References
    ----------
    https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py
    """
    if x.size < 3:
        return np.array([], dtype=int)
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    rising_edges = np.where((np.hstack((dx, 0)) <= 0) &
                            (np.hstack((0, dx)) > 0))[0]
    falling_edges = np.where((np.hstack((dx, 0)) < 0) &
                             (np.hstack((0, dx)) >= 0))[0]
    indices = np.unique(np.hstack((rising_edges, falling_edges)))
    if indices.size and min_height > 0:
        indices = indices[x[indices] >= min_height]
    return indices


class MultimodalChromatogramShapeFitter(ChromatogramShapeFitterBase):
    def __init__(self, chromatogram, max_peaks=5, smooth=True, fitter=BiGaussianModel()):
        super(MultimodalChromatogramShapeFitter, self).__init__(chromatogram, smooth=smooth, fitter=fitter)
        self.max_peaks = max_peaks
        self.params_list = []
        self.params_dict_list = []

        if self.is_invalid():
            self.handle_invalid()
        else:
            try:
                self.extract_arrays()
                self.peak_shape_fit()
                self.perform_line_test()
            except TypeError:
                self.handle_invalid()

    @property
    def fit_parameters(self):
        return self.params_dict_list

    def __repr__(self):
        return "MultimodalChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)

    def peak_shape_fit(self):
        return self.set_up_peak_fit()

    def set_up_peak_fit(self, ys=None, min_height=0, peak_count=0):
        xs = self.xs
        if ys is None:
            ys = self.ys
        params = self.shape_fitter.guess(xs, ys)
        params_dict = self.shape_fitter.params_to_dict(params)

        indices = peak_indices(ys, min_height)
        center = xs[max(indices, key=lambda x: ys[x])]
        params_dict['center'] = center

        fit = leastsq(self.shape_fitter.fit,
                      params_dict.values(), (xs, ys))
        params = fit[0]
        params_dict = FittedPeakShape(self.shape_fitter.params_to_dict(params), self.shape_fitter)
        self.params_list.append(params)
        self.params_dict_list.append(params_dict)

        residuals = self.shape_fitter.fit(params, xs, ys)

        fitted_apex_index = search.get_nearest(xs, params_dict['center'], 0)
        fitted_apex = ys[fitted_apex_index]

        new_min_height = fitted_apex * 0.5

        if new_min_height < min_height:
            min_height *= 0.85
        else:
            min_height = new_min_height

        indices = peak_indices(residuals, min_height)

        peak_count += 1
        if indices.size and peak_count < self.max_peaks:
            residuals, params_dict = self.set_up_peak_fit(residuals, min_height, peak_count=peak_count)

        return residuals, params_dict

    def compute_fitted(self):
        xs = self.xs
        fitted = np.zeros_like(xs)
        for params_dict in self.params_dict_list:
            fitted += self.shape_fitter.shape(xs, **params_dict)
        return fitted

    def compute_residuals(self):
        return self.ys - self.compute_fitted()

    def iterfits(self):
        xs = self.xs
        for params_dict in self.params_dict_list:
            yield self.shape_fitter.shape(xs, **params_dict)


class AdaptiveMultimodalChromatogramShapeFitter(ChromatogramShapeFitterBase):
    def __init__(self, chromatogram, max_peaks=5, smooth=True, fitters=None):
        if fitters is None:
            fitters = (GaussianModel(), BiGaussianModel(), PenalizedSkewedGaussianModel(),)
        super(AdaptiveMultimodalChromatogramShapeFitter, self).__init__(
            chromatogram, smooth=smooth, fitter=fitters[0])
        self.max_peaks = max_peaks
        self.fitters = fitters
        self.params_list = []
        self.params_dict_list = []

        self.alternative_fits = []
        self.best_fit = None

        if self.is_invalid():
            self.handle_invalid()
        else:
            self.extract_arrays()
            self.peak_shape_fit()
            self.perform_line_test()

    def is_invalid(self):
        return len(self.chromatogram) < MIN_POINTS

    @property
    def fit_parameters(self):
        return self.best_fit.fit_parameters

    @property
    def xs(self):
        try:
            return self.best_fit.xs
        except AttributeError:
            return self._xs

    @xs.setter
    def xs(self, value):
        self._xs = value

    @property
    def ys(self):
        try:
            return self.best_fit.ys
        except AttributeError:
            return self._ys

    @ys.setter
    def ys(self, value):
        self._ys = value

    def compute_fitted(self):
        return self.best_fit.compute_fitted()

    def compute_residuals(self):
        return self.best_fit.compute_residuals()

    def _get_gap_size(self):
        return np.average(self.xs[1:] - self.xs[:-1],
                          weights=(self.ys[1:] + self.ys[:-1])) * 2

    def has_sparse_tails(self):
        gap = self._get_gap_size()
        partition = [False, False]
        if self.xs[1] - self.xs[0] > gap:
            partition[0] = True
        if self.xs[-1] - self.xs[-2] > gap:
            partition[1] = True
        return partition

    def generate_trimmed_chromatogram_slices(self):
        for tails in set(product(*zip(self.has_sparse_tails(), [False, False]))):
            if not tails[0] and not tails[1]:
                continue
            if tails[0]:
                slice_start = self.xs[1]
            else:
                slice_start = self.xs[0]
            if tails[1]:
                slice_end = self.xs[-2]
            else:
                slice_end = self.xs[-1]
            subset = self.chromatogram.slice(slice_start, slice_end)
            yield subset

    def peak_shape_fit(self):
        for fitter in self.fitters:
            model_fit = ProfileSplittingMultimodalChromatogramShapeFitter(
                self.chromatogram, self.max_peaks, self.smooth, fitter=fitter)
            self.alternative_fits.append(model_fit)
            model_fit = MultimodalChromatogramShapeFitter(
                self.chromatogram, self.max_peaks, self.smooth, fitter=fitter)
            self.alternative_fits.append(model_fit)
            for subset in self.generate_trimmed_chromatogram_slices():
                model_fit = ProfileSplittingMultimodalChromatogramShapeFitter(
                    subset, self.max_peaks,
                    self.smooth, fitter=fitter)
                self.alternative_fits.append(model_fit)
        ix = np.nanargmin([f.line_test for f in self.alternative_fits])
        # self.best_fit = min(self.alternative_fits, key=lambda x: x.line_test)
        self.best_fit = self.alternative_fits[ix]
        self.params_list = self.best_fit.params_list
        self.params_dict_list = self.best_fit.params_dict_list
        self.shape_fitter = self.best_fit.shape_fitter

    def perform_line_test(self):
        self.line_test = self.best_fit.line_test

    def plot(self, *args, **kwargs):
        return self.best_fit.plot(*args, **kwargs)

    def iterfits(self):
        xs = self.xs
        for params_dict in self.params_dict_list:
            yield self.shape_fitter.shape(xs, **params_dict)

    def __repr__(self):
        return "AdaptiveMultimodalChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)


class SplittingPoint(object):
    __slots__ = ["first_maximum", "minimum", "second_maximum", "minimum_index", "total_distance"]

    def __init__(self, first_maximum, minimum, second_maximum, minimum_index):
        self.first_maximum = first_maximum
        self.minimum = minimum
        self.second_maximum = second_maximum
        self.minimum_index = minimum_index
        self.total_distance = self.compute_distance()

    def compute_distance(self):
        return (self.first_maximum - self.minimum) + (self.second_maximum - self.minimum)

    def __repr__(self):
        return "SplittingPoint(%0.4f, %0.4f, %0.4f, %0.2f, %0.3e)" % (
            self.first_maximum, self.minimum, self.second_maximum, self.minimum_index, self.total_distance)


class ProfileSplittingMultimodalChromatogramShapeFitter(ChromatogramShapeFitterBase):
    def __init__(self, chromatogram, max_splits=3, smooth=True, fitter=BiGaussianModel()):
        super(ProfileSplittingMultimodalChromatogramShapeFitter, self).__init__(
            chromatogram, smooth=smooth, fitter=fitter)
        self.max_splits = max_splits
        self.params_list = []
        self.params_dict_list = []
        self.partition_sites = []

        if self.is_invalid():
            self.handle_invalid()
        else:
            self.extract_arrays()
            self.peak_shape_fit()
            self.perform_line_test()

    def __repr__(self):
        return "ProfileSplittingMultimodalChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)

    def _extreme_indices(self, ys):
        maxima_indices = peak_indices(ys)
        minima_indices = peak_indices(-ys)
        return maxima_indices, minima_indices

    def locate_extrema(self, xs=None, ys=None):
        if xs is None:
            xs = self.xs
        if ys is None:
            ys = self.ys

        maxima_indices, minima_indices = self._extreme_indices(ys)
        candidates = []

        for i in range(len(maxima_indices)):
            max_i = maxima_indices[i]
            for j in range(i + 1, len(maxima_indices)):
                max_j = maxima_indices[j]
                for k in range(len(minima_indices)):
                    min_k = minima_indices[k]
                    y_i = ys[max_i]
                    y_j = ys[max_j]
                    y_k = ys[min_k]
                    if max_i < min_k < max_j and (y_i - y_k) > (y_i * 0.01) and (
                            y_j - y_k) > (y_j * 0.01):
                        point = SplittingPoint(y_i, y_k, y_j, xs[min_k])
                        candidates.append(point)
        if candidates:
            best_point = max(candidates, key=lambda x: x.total_distance)
            self.partition_sites.append(best_point)

        return candidates

    def build_partitions(self):
        segments = []

        last_x = self.xs.min() - 1
        for point in self.partition_sites:
            mask = (self.xs <= point.minimum_index) & (self.xs > last_x)
            if any(mask):
                xs, ys = self.xs[mask], self.ys[mask]
                # if len(xs) > 1:
                segments.append((xs, ys))
            last_x = point.minimum_index
        mask = self.xs > last_x
        if any(mask):
            xs, ys = self.xs[mask], self.ys[mask]
            # if len(xs) > 1:
            segments.append((xs, ys))
        return segments

    def set_up_peak_fit(self, xs, ys):
        params = self.shape_fitter.guess(xs, ys)
        params_dict = FittedPeakShape(self.shape_fitter.params_to_dict(params), self.shape_fitter)
        if len(params) > len(xs):
            self.params_list.append(params)
            self.params_dict_list.append(params_dict)
            return ys, params_dict

        fit = leastsq(self.shape_fitter.fit,
                      params_dict.values(), (xs, ys))
        params = fit[0]
        params_dict = FittedPeakShape(self.shape_fitter.params_to_dict(params), self.shape_fitter)
        self.params_list.append(params)
        self.params_dict_list.append(params_dict)

        residuals = self.shape_fitter.fit(params, xs, ys)
        return residuals, params_dict

    def peak_shape_fit(self):
        self.locate_extrema()
        for segment in self.build_partitions():
            self.set_up_peak_fit(*segment)

    def compute_fitted(self):
        fitted = []
        for segment, params_dict in zip(self.build_partitions(), self.params_dict_list):
            fitted.append(self.shape_fitter.shape(segment[0], **params_dict))
        return np.concatenate(fitted)

    def compute_residuals(self):
        return self.ys - self.compute_fitted()

    def iterfits(self):
        for segment, params_dict in zip(self.build_partitions(), self.params_dict_list):
            yield self.shape_fitter.shape(segment[0], **params_dict)


try:
    _bigaussian_shape = bigaussian_shape
    _skewed_gaussian_shape = skewed_gaussian_shape
    _gaussian_shape = gaussian_shape

    from ms_deisotope._c.feature_map.profile_transform import (
        bigaussian_shape, skewed_gaussian_shape, gaussian_shape)

    has_c = True
except ImportError:
    has_c = False
