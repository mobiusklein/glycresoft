'''Extra logic for fitting a model across multiple LC-MS runs.
'''
from collections import defaultdict

import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


import glycopeptidepy

from .model import AbundanceWeightedPeptideFactorElutionTimeFitter, make_counter
from .structure import CommonGlycopeptide
from .linear_regression import weighted_linear_regression_fit


class ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter(AbundanceWeightedPeptideFactorElutionTimeFitter):
    def __init__(self, chromatograms, factors=None, scale=1, replicate_key_attr='analysis_name',
                 use_retention_time_normalization=False, reference_sample=None):
        if replicate_key_attr is None:
            replicate_key_attr = 'analysis_name'
        if factors is None:
            factors = ['Hex', 'HexNAc', 'Fuc', 'Neu5Ac']
        self.replicate_key_attr = replicate_key_attr
        self.reference_sample = reference_sample
        self._replicate_to_indicator = defaultdict(make_counter(0))
        self.use_retention_time_normalization = use_retention_time_normalization
        self.run_normalizer = None
        # Ensure that _replicate_to_indicator is properly initialized
        # for obs in chromatograms:
        #     _ = self._replicate_to_indicator[self._get_replicate_key(obs)]
        self.chromatograms = chromatograms

        index, samples = self._build_common_glycopeptides()
        self.replicate_names = samples
        if self.use_retention_time_normalization:
            self.run_normalizer = LinearRetentionTimeCorrector(index, samples, self.reference_sample)
            if self.reference_sample is None:
                self.reference_sample = self.run_normalizer.reference_key
            self.run_normalizer.fit()
        else:
            if self.reference_sample is None:
                self.reference_sample = samples[0]
        _ = self._replicate_to_indicator[self.reference_sample]
        for sample_key in samples:
            _ = self._replicate_to_indicator[sample_key]
        super(ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter, self).__init__(
            chromatograms, list(factors), scale)

    def reset_reference_sample(self, reference_sample):
        self.reference_sample = reference_sample
        self._replicate_to_indicator = defaultdict(make_counter(0))
        _ = self._replicate_to_indicator[self.reference_sample]
        for sample_key in self.replicate_names:
            _ = self._replicate_to_indicator[sample_key]
        self.data = self._prepare_data_matrix(self.neutral_mass_array)

    def _get_apex_time(self, chromatogram):
        value = super(ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter,
                      self)._get_apex_time(chromatogram)
        if self.use_retention_time_normalization and self.run_normalizer is not None:
            replicate_id = self._get_replicate_key(chromatogram)
            value = self.run_normalizer.correct(value, replicate_id)
        return value

    def _build_common_glycopeptides(self):
        index = defaultdict(dict)
        for case in self.chromatograms:
            index[str(case.structure)][self._get_replicate_key(case)] = case
        result = [CommonGlycopeptide(glycopeptidepy.parse(k), v) for k, v in index.items()]
        samples = set()
        for case in result:
            samples.update(case.keys())
        return result, sorted(samples)

    def _get_replicate_key(self, chromatogram):
        if isinstance(self.replicate_key_attr, (str, unicode)):
            return getattr(chromatogram, self.replicate_key_attr)
        elif callable(self.replicate_key_attr):
            return self.replicate_key_attr(chromatogram)

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

    def _prepare_data_vector(self, chromatogram, no_intercept=False):
        data_vector = super(
            ReplicatedAbundanceWeightedPeptideFactorElutionTimeFitter,
            self)._prepare_data_vector(chromatogram, no_intercept=no_intercept)
        p = len(self._replicate_to_indicator)
        replicates = [0 for _ in range(p)]
        indicator = dict(self._replicate_to_indicator)
        if not no_intercept:
            try:
                # Here, if one of the levels is not omitted, the matrix will be linearly dependent.
                # So drop the 0th factor level.
                j = self._get_replicate_key(chromatogram)
                if j != 0:
                    replicates[indicator[j]] = 1
            except KeyError:
                import warnings
                warnings.warn(
                    "Replicate Key %s not part of the model." % (j, ))
        return np.hstack((replicates, data_vector))


class LinearRetentionTimeCorrector(object):
    def __init__(self, common_observations, sample_keys, reference_key=None):
        self.common_observations = common_observations
        self.sample_keys = sample_keys
        self.reference_key = reference_key
        self.sample_to_correction_parameters = dict()
        self.sample_to_vector = dict()
        if self.reference_key is None:
            self.reference_key = self.select_reference()

    def build_sample_distance_matrix(self):
        n = len(self.sample_keys)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            ivec = self._make_vector_for(self.sample_keys[i])
            for j in range(i + 1, n):
                jvec = self._make_vector_for(self.sample_keys[j])
                wls = self._fit_correction(jvec, ivec)
                distance = wls.R2
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def select_reference(self):
        distance_matrix = self.build_sample_distance_matrix()
        return self.sample_keys[np.argmin(distance_matrix.sum(axis=0))]

    def _make_vector_for(self, sample):
        if sample in self.sample_to_vector:
            return self.sample_to_vector[sample]
        result = []
        for obs in self.common_observations:
            try:
                result.append(obs[sample].apex_time)
            except KeyError:
                result.append(np.nan)
        result = self.sample_to_vector[sample] = np.array(result)
        return result

    def _fit_correction(self, a, b):
        delta = b - a
        mask = ~(np.isnan(delta))
        wls_fit = weighted_linear_regression_fit(
            b[mask], delta[mask], prepare=True)
        return wls_fit

    def fit_correction_for(self, sample_key):
        sample = self._make_vector_for(sample_key)
        reference = self._make_vector_for(self.reference_key)
        wls_fit = self._fit_correction(sample, reference)
        self.sample_to_correction_parameters[sample_key] = wls_fit
        return wls_fit

    def fit(self):
        for sample_key in self.sample_keys:
            if sample_key == self.reference_key:
                continue
            self.fit_correction_for(sample_key)

    def correct(self, x, sample_key):
        return x - self._correction_factor_for(x, sample_key)

    def _correction_factor_for(self, x, sample_key):
        if sample_key == self.reference_key:
            return 0
        fit = self.sample_to_correction_parameters[sample_key]
        return (fit.parameters[1] * x + fit.parameters[0])

    def plot(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1)
        X = self._make_vector_for(self.reference_key)
        mask = ~np.isnan(X)
        X = X[mask]
        Xhat = np.arange(X.min(), X.max())
        for sample_key in sorted(self.sample_keys):
            if sample_key == self.reference_key:
                ax.scatter(X, np.zeros_like(X), label="%s Reference" % (sample_key, ))
            else:
                fit = self.sample_to_correction_parameters[sample_key]
                ax.scatter(fit.data[0][:, 1], fit.data[1],
                        label=r'%s %0.2fX + %0.2f ($R^2$=%0.2f)' % (
                            sample_key, fit.parameters[1], fit.parameters[0], fit.R2))
                Yhat = Xhat * fit.parameters[1] + fit.parameters[0]
                ax.plot(Xhat, Yhat, linestyle='--')
        legend = ax.legend(fontsize=6)
        frame = legend.get_frame()
        frame.set_linewidth(0.5)
        frame.set_alpha(0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\delta$ Time")
        return ax
