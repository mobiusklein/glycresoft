import numpy as np
from scipy.optimize import leastsq
from numpy import pi, sqrt, exp
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

from ms_peak_picker import search


def skewgauss(xs, center, amplitude, sigma, gamma):
    norm = (amplitude) / (sigma * sqrt(2 * pi)) * exp(-((xs - center) ** 2) / (2 * sigma ** 2))
    skew = (1 + erf((gamma * (xs - center)) / (sigma * sqrt(2))))
    return norm * skew


def fit_skewgauss(params, xs, ys):
    center, amplitude, sigma, gamma = params
    return ys - skewgauss(xs, center, amplitude, sigma, gamma) * (sigma / 2. if sigma > 2 else 1.)


def params_to_dict(params):
    center, amplitude, sigma, gamma = params
    return dict(center=center, amplitude=amplitude, sigma=sigma, gamma=gamma)


class ChromatogramShapeFitter(object):
    def __init__(self, chromatogram, smooth=True, fitter=fit_skewgauss):
        self.chromatogram = chromatogram
        self.smooth = smooth
        self.params = None
        self.params_dict = None
        self.xs = None
        self.ys = None
        self.line_test = None
        self.off_center = None
        self.shape_fitter = fitter

        if len(chromatogram) < 5:
            self.handle_invalid()
        else:
            self.extract_arrays()
            self.peak_shape_fit()
            self.perform_line_test()
            self.off_center_factor()

    def __repr__(self):
        return "ChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)

    def handle_invalid(self):
        self.line_test = 0.0

    def off_center_factor(self):
        self.off_center = abs(1 - abs(1 - (2 * abs(
            self.xs[self.ys.argmax()] - self.params_dict['center']) / abs(self.params_dict['sigma']))))
        if self.off_center > 1:
            self.off_center = 1. / self.off_center
        self.line_test /= self.off_center

    def extract_arrays(self):
        self.xs, self.ys = self.chromatogram.as_arrays()
        if self.smooth:
            self.ys = gaussian_filter1d(self.ys, 1)

    def peak_shape_fit(self):
        xs, ys = self.xs, self.ys
        center = np.average(xs, weights=ys / ys.sum())
        height_at = np.abs(xs - center).argmin()
        apex = ys[height_at]
        sigma = np.abs(center - xs[[search.nearest_left(ys, apex / 2, height_at),
                                    search.nearest_right(ys, apex / 2, height_at + 1)]]).sum()
        gamma = 1
        fit = leastsq(self.shape_fitter, (center, apex, sigma, gamma), (xs, ys))
        params = fit[0]
        self.params = params
        self.params_dict = params_to_dict(params)

    def perform_line_test(self):
        xs, ys = self.xs, self.ys
        residuals = self.shape_fitter(self.params, xs, ys)
        line_test = (residuals ** 2).sum() / (
            ((ys - ((ys.max() + ys.min()) / 2.)) ** 2).sum())
        self.line_test = line_test


def shape_fit_test(chromatogram, smooth=True):
    return ChromatogramShapeFitter(chromatogram, smooth).line_test
