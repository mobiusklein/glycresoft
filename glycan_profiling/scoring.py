from collections import namedtuple, defaultdict

import numpy as np
from scipy.optimize import leastsq
from numpy import pi, sqrt, exp
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

from ms_peak_picker import search
from ms_deisotope.scoring import g_test_scaled
from ms_deisotope.averagine import glycan, Averagine, PROTON
from ms_peak_picker.peak_set import FittedPeak

from .trace import total_intensity

epsilon = 1e-6


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
    def __init__(self, chromatogram, smooth=True):
        self.chromatogram = chromatogram
        self.smooth = smooth
        self.params = None
        self.params_dict = None
        self.xs = None
        self.ys = None
        self.line_test = None
        self.off_center = None

        self.extract_arrays()
        self.peak_shape_fit()
        self.perform_line_test()
        self.off_center_factor()

    def __repr__(self):
        return "ChromatogramShapeFitter(%s, %0.4f)" % (self.chromatogram, self.line_test)

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
        fit = leastsq(fit_skewgauss, (center, apex, sigma, gamma), (xs, ys))
        params = fit[0]
        self.params = params
        self.params_dict = params_to_dict(params)

    def perform_line_test(self):
        xs, ys = self.xs, self.ys
        residuals = fit_skewgauss(self.params, xs, ys)
        line_test = (residuals ** 2).sum() / (
            ((ys - ((ys.max() + ys.min()) / 2.)) ** 2).sum())
        self.line_test = line_test


class ChromatogramSpacingFitter(object):
    def __init__(self, chromatogram):
        self.chromatogram = chromatogram
        self.rt_deltas = []
        self.intensity_deltas = []
        self.score = None

        self.fit()

    def fit(self):
        intensities = map(total_intensity, self.chromatogram.peaks)
        last_rt = self.chromatogram.retention_times[0]
        last_int = intensities[0]

        for rt, inten in zip(self.chromatogram.retention_times[1:], intensities[1:]):
            d_rt = rt - last_rt
            self.rt_deltas.append(d_rt)
            self.intensity_deltas.append(abs(last_int - inten))
            last_rt = rt
            last_int = inten

        self.rt_deltas = np.array(self.rt_deltas)
        self.intensity_deltas = np.array(self.intensity_deltas)

        self.score = np.average(self.rt_deltas, weights=self.intensity_deltas / self.intensity_deltas.sum())

    def __repr__(self):
        return "ChromatogramSpacingFitter(%s, %0.4f)" % (self.chromatogram, self.score)


def shape_fit_test(chromatogram, smooth=True):
    return ChromatogramShapeFitter(chromatogram, smooth).line_test


def envelope_to_peak_list(envelope):
    return [FittedPeak(e[0], e[1], 0, 0, 0, 0, 0, 0, 0) for e in envelope]


def scale_theoretical_isotopic_pattern(eid, tid):
    total = sum(p.intensity for p in eid)
    for p in tid:
        p.intensity *= total


def score_isotopic_pattern_averagine(deconvoluted_peak, averagine=glycan, charge_carrier=PROTON):
    eid = envelope_to_peak_list(deconvoluted_peak.envelope)
    tid = averagine.isotopic_cluster(
        deconvoluted_peak.mz, deconvoluted_peak.charge,
        charge_carrier=charge_carrier)
    scale_theoretical_isotopic_pattern(eid, tid)
    tid = align_peak_list(eid, tid)
    return g_test_scaled(None, eid, tid)


def get_nearest_index(query_mz, peak_list):
    best_index = None
    best_error = float('inf')

    for i, peak in enumerate(peak_list):
        error = abs(peak.mz - query_mz)
        if error < best_error:
            best_error = error
            best_index = i
    return best_index


def align_peak_list(experimental, theoretical):
    retain = []
    for peak in experimental:
        retain.append(theoretical[get_nearest_index(peak.mz, theoretical)])
    return retain


def chromatogram_to_peaks(chromatogram):
    for scan in chromatogram.peaks:
        for peak in scan:
            yield peak


def evaluate_chromatogram_isotopic_fit(chromatogram, averagine=glycan, charge_carrier=PROTON):
    scores = []
    intensity = []
    for scan in chromatogram.peaks:
        for peak in scan:
            score = score_isotopic_pattern_averagine(peak, averagine=averagine, charge_carrier=charge_carrier)
            scores.append(score)
            intensity.append(peak.intensity)
    intensity = np.array(intensity)
    scores = np.array(scores)
    return np.average(scores, weights=intensity / intensity.sum())


class IsotopicPatternConsistencyFitter(object):
    def __init__(self, chromatogram, averagine=glycan, charge_carrier=PROTON):
        self.chromatogram = chromatogram
        self.averagine = averagine
        self.charge_carrier = charge_carrier
        self.scores = []
        self.intensity = []
        self.mean_fit = None

        self.fit()

    def __repr__(self):
        return "IsotopicPatternConsistencyFitter(%s, %0.4f)" % (self.chromatogram, self.mean_fit)

    def fit(self):
        for scan in self.chromatogram.peaks:
            for peak in scan:
                score = score_isotopic_pattern_averagine(
                    peak, averagine=self.averagine, charge_carrier=self.charge_carrier)
                self.scores.append(score)
                self.intensity.append(peak.intensity)
        self.intensity = np.array(self.intensity)
        self.scores = np.array(self.scores)
        self.mean_fit = np.average(self.scores, weights=self.intensity / self.intensity.sum())


class ChargeStateDistributionScoringModelBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, chromatogram, *args, **kwargs):
        return 0


class UniformChargeStateScoringModel(ChargeStateDistributionScoringModelBase):
    def score(self, chromatogram, *args, **kwargs):
        return min(0.4 * chromatogram.n_charge_states, 1.0)


def ones(x):
    return (x - (np.floor(x / 10.) * 10))


def neighborhood_of(x):
    n = x / 100.
    up = ones(n) > 5
    if up:
        neighborhood = (np.floor(n / 10.) + 1) * 10
    else:
        neighborhood = (np.floor(n / 10.) + 1) * 10
    return neighborhood * 100


class NegativeModeMassScalingChargeStateScoringModel(ChargeStateDistributionScoringModelBase):
    def __init__(self, table):
        self.table = table

    def score(self, chromatogram, *args, **kwargs):
        total = 0.
        neighborhood = neighborhood_of(chromatogram.neutral_mass)
        if neighborhood not in self.table:
            import warnings
            warnings.warn(
                ("%f is too large for this charge state "
                 "scoring model. Defaulting to uniform model") % neighborhood)
            return UniformChargeStateScoringModel().score(chromatogram, *args, **kwargs)
        bins = self.table[neighborhood]

        for charge in chromatogram.charge_states:
            total += bins[charge]
        return total

    @classmethod
    def fit(cls, observations, missing=0.01):
        bins = defaultdict(lambda: defaultdict(float))

        for sol in observations:
            neighborhood = neighborhood_of(sol.neutral_mass)
            for c in sol.charge_states:
                bins[neighborhood][c] += 1

        model_table = {}

        all_states = set()
        for level in bins.values():
            all_states.update(level.keys())

        for neighborhood, counts in bins.items():
            for c in all_states:
                if counts[c] == 0:
                    counts[c] = missing
            total = sum(counts.values())
            entry = {k: v / total for k, v in counts.items()}
            model_table[neighborhood] = entry

        return cls(model_table)


def score_chromatogram(chromatogram):
    line_score = max(1 - ChromatogramShapeFitter(chromatogram).line_test, epsilon)
    isotopic_fit = max(1 - IsotopicPatternConsistencyFitter(chromatogram).mean_fit, epsilon)
    spacing_fit = max(1 - ChromatogramSpacingFitter(chromatogram).score * 2, epsilon)
    charge_count = UniformChargeStateScoringModel().score(chromatogram)
    return (line_score * isotopic_fit * spacing_fit * charge_count)


scores = namedtuple("scores", ["line_score", "isotopic_fit", "spacing_fit", "charge_count"])


def score_chromatogram2(chromatogram):
    line_score = max(1 - ChromatogramShapeFitter(chromatogram).line_test, epsilon)
    isotopic_fit = max(1 - IsotopicPatternConsistencyFitter(chromatogram).mean_fit, epsilon)
    spacing_fit = max(1 - ChromatogramSpacingFitter(chromatogram).score * 2, epsilon)
    charge_count = UniformChargeStateScoringModel().score(chromatogram)
    return scores(line_score, isotopic_fit, spacing_fit, charge_count)


class NetworkScoreDistributor(object):
    def __init__(self, solutions, network):
        self.solutions = solutions
        self.network = network

    def distribute(self, base_coef=0.8, support_coef=0.2):
        solution_map = {
            sol.chromatogram.composition: sol
            for sol in self.solutions
            if sol.chromatogram.composition is not None
        }

        cg = self.network

        for sol in self.solutions:
            if sol.composition is not None:
                cn = cg[sol.composition]
                base = base_coef * sol.score
                support = 0
                for edge in cn.edges:
                    other = edge[cn]
                    if other in solution_map:
                        other_sol = solution_map[other]
                        support += support_coef * edge.weight * other_sol.score
                sol.score = base + min(support, support_coef)
            else:
                sol.score *= base_coef


class ChromatogramSolution(object):
    def __init__(self, chromatogram, score=None):
        self.chromatogram = chromatogram
        self.score = score

        if score is None:
            self.compute_score()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "chromatogram"), name)

    def __len__(self):
        return len(self.chromatogram)

    def __iter__(self):
        return iter(self.chromatogram)

    def compute_score(self):
        self.score = score_chromatogram(self.chromatogram)

    def __repr__(self):
        return "ChromatogramSolution(%s, %0.4f, %d, %0.4f)" % (
            self.chromatogram.composition, self.chromatogram.neutral_mass,
            self.chromatogram.n_charge_states, self.score)
