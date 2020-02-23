# -*- coding: utf8 -*-
'''This module implements techniques derived from the pGlyco2
FDR estimation procedure described in:

[1] Liu, M.-Q., Zeng, W.-F., Fang, P., Cao, W.-Q., Liu, C., Yan, G.-Q., … Yang, P.-Y.
    (2017). pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality
    control and one-step mass spectrometry for intact glycopeptide identification.
    Nature Communications, 8(1), 438. https://doi.org/10.1038/s41467-017-00535-2
[2] Zeng, W.-F., Liu, M.-Q., Zhang, Y., Wu, J.-Q., Fang, P., Peng, C., … Yang, P. (2016).
    pGlyco: a pipeline for the identification of intact N-glycopeptides by using HCD-
    and CID-MS/MS and MS3. Scientific Reports, 6(April), 25102. https://doi.org/10.1038/srep25102
'''
import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass


from glypy.utils import Enum

from glycan_profiling.task import TaskBase

from glycan_profiling.tandem.target_decoy import NearestValueLookUp, TargetDecoyAnalyzer
from glycan_profiling.tandem.spectrum_match import SpectrumMatch
from glycan_profiling.tandem.glycopeptide.core_search import approximate_internal_size_of_glycan

from .mixture import GammaMixture, GaussianMixtureWithPriorComponent


def noop(*args, **kwargs):
    pass


class GlycopeptideFDREstimationStrategy(Enum):
    multipart_gamma_gaussian_mixture = 0
    peptide_fdr = 1
    glycan_fdr = 2
    glycopeptide_fdr = 3

GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture.add_name("multipart")
GlycopeptideFDREstimationStrategy.peptide_fdr.add_name("peptide")
GlycopeptideFDREstimationStrategy.glycan_fdr.add_name("glycan")


class FiniteMixtureModelFDREstimator(object):
    def __init__(self, decoy_scores, target_scores):
        self.decoy_scores = np.array(decoy_scores)
        self.target_scores = np.array(target_scores)
        self.decoy_mixture = None
        self.target_mixture = None
        self.fdr_map = None

    def log(self, message):
        print(message)

    def estimate_gamma(self, max_components=10):
        models = []
        bics = []
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.decoy_mixture = GammaMixture([1.0], [1.0], [1.0])
            return self.decoy_mixture
        for i in range(1, max_components + 1):
            self.log("Fitting %d Components" % (i,))
            model = GammaMixture.fit(self.decoy_scores, i)
            bic = model.bic(self.decoy_scores)
            models.append(model)
            bics.append(bic)
            self.log("BIC: %g" % (bic,))
        i = np.argmin(bics)
        self.log("Selected %d Components" % (i + 1,))
        self.decoy_mixture = models[i]
        return self.decoy_mixture

    def estimate_gaussian(self, max_components=10):
        models = []
        bics = []
        n = len(self.target_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few target observations")
            self.target_mixture = GaussianMixtureWithPriorComponent([1.0], [1.0], self.decoy_mixture, [0.5, 0.5])
            return self.target_mixture
        for i in range(1, max_components + 1):
            self.log("Fitting %d Components" % (i,))
            model = GaussianMixtureWithPriorComponent.fit(
                self.target_scores, i, self.decoy_mixture, deterministic=True)
            bic = model.bic(self.target_scores)
            models.append(model)
            bics.append(bic)
            self.log("BIC: %g" % (bic,))
        i = np.argmin(bics)
        self.log("Selected %d Components" % (i + 1,))
        self.target_mixture = models[i]
        return self.target_mixture

    def estimate_posterior_error_probability(self, X):
        return self.target_mixture.prior.score(X) * self.target_mixture.weights[
            -1] / self.target_mixture.score(X)

    def estimate_fdr(self, X):
        X_ = np.array(sorted(X, reverse=True))
        pep = self.estimate_posterior_error_probability(X_)
        # The FDR is the expected value of PEP, or the average PEP in this case.
        # The expression below is a cumulative mean (the cumulative sum divided
        # by the number of elements in the sum)
        fdr = np.cumsum(pep) / np.arange(1, len(X_) + 1)
        # Use searchsorted on the ascending ordered version of X_
        # to find the indices of the origin values of X, then map
        # those into the ascending ordering of the FDR vector to get
        # the FDR estimates of the original X
        fdr[np.isnan(fdr)] = 1.0
        fdr_descending = fdr[::-1]
        for i in range(1, fdr_descending.shape[0]):
            if fdr_descending[i - 1] < fdr_descending[i]:
                fdr_descending[i] = fdr_descending[i - 1]
        fdr = fdr_descending[::-1]
        fdr = fdr[::-1][np.searchsorted(X_[::-1], X)]
        return fdr

    def plot_mixture(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = np.arange(1, max(self.target_scores), 0.1)
        ax.plot(X,
                np.exp(self.target_mixture.logpdf(X)).sum(axis=1))
        for col in np.exp(self.target_mixture.logpdf(X)).T:
            ax.plot(X, col, linestyle='--')
        ax.hist(self.target_scores, bins=100, density=1, alpha=0.15)
        return ax

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        points = np.linspace(
            min(self.target_scores.min(), self.decoy_scores.min()),
            max(self.target_scores.max(), self.decoy_scores.max()),
            10000)
        target_scores = np.sort(self.target_scores)
        target_counts = [(self.target_scores >= i).sum() for i in points]
        decoy_counts = [(self.decoy_scores >= i).sum() for i in points]
        fdr = self.estimate_fdr(target_scores)
        at_5_percent = np.where(fdr < 0.05)[0][0]
        at_1_percent = np.where(fdr < 0.01)[0][0]
        line1 = ax.plot(points, target_counts, label='Target', color='blue')
        line2 = ax.plot(points, decoy_counts, label='Decoy', color='orange')
        ax.vlines(target_scores[at_5_percent], 0, np.max(target_counts), linestyle='--', color='blue', lw=0.75)
        ax.vlines(target_scores[at_1_percent], 0, np.max(target_counts), linestyle='--', color='blue', lw=0.75)
        ax2 = ax.twinx()
        line3 = ax2.plot(target_scores, fdr, label='FDR', color='grey', linestyle='--')
        ax.legend([line1[0], line2[0], line3[0]], ['Target', 'Decoy', 'FDR'])
        return ax

    def fit(self, max_components=10):
        self.estimate_gamma(max_components)
        self.estimate_gaussian(max_components)
        fdr = self.estimate_fdr(self.target_scores)
        self.fdr_map = NearestValueLookUp(zip(self.target_scores, fdr))
        return self.fdr_map


class GlycanSizeCalculator(object):

    def __init__(self):
        self.glycan_size_cache = dict()

    def get_internal_size(self, glycan_composition):
        key = str(glycan_composition)
        if key in self.glycan_size_cache:
            return self.glycan_size_cache[key]
        n = approximate_internal_size_of_glycan(glycan_composition)
        self.glycan_size_cache[key] = n
        return n

    def __call__(self, glycan_composition):
        return self.get_internal_size(glycan_composition)


class GlycopeptideFDREstimator(TaskBase):
    display_fields = False

    def __init__(self, groups, strategy=None):
        if strategy is None:
            strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
        else:
            strategy = GlycopeptideFDREstimationStrategy[strategy]
        self.strategy = strategy
        self.grouper = groups
        self.glycan_fdr = None
        self.peptide_fdr = None
        self.glycopeptide_fdr = None

    def fit_glycan_fdr(self):
        tt_gpsms = self.grouper.exclusive_match_groups['target_peptide_target_glycan']
        td_gpsms = self.grouper.exclusive_match_groups['target_peptide_decoy_glycan']

        target_glycan_scores = np.array(
            [s.score_set.glycan_score for s in tt_gpsms])
        decoy_glycan_scores = np.array(
            [s.score_set.glycan_score for s in td_gpsms])

        sizer = GlycanSizeCalculator()
        size_mask = np.array(
            [sizer(s.target.glycan_composition)
             for s in td_gpsms], dtype=int)

        glycan_fdr = FiniteMixtureModelFDREstimator(
            decoy_glycan_scores[(size_mask > 3) & (decoy_glycan_scores > 1)],
            target_scores=target_glycan_scores[target_glycan_scores > 1])
        glycan_fdr.log = noop
        glycan_fdr.fit()

        glycan_fdr_mapping = NearestValueLookUp(zip(glycan_fdr.estimate_fdr(glycan_fdr.target_scores),
                                                    glycan_fdr.target_scores))
        self.log("5%% Glycan FDR = %f (%d)" % (glycan_fdr_mapping[0.05], (
            glycan_fdr.target_scores > glycan_fdr_mapping[0.05]).sum()))
        self.log("1%% Glycan FDR = %f (%d)" % (glycan_fdr_mapping[0.01], (
            glycan_fdr.target_scores > glycan_fdr_mapping[0.01]).sum()))
        self.glycan_fdr = glycan_fdr
        return self.glycan_fdr

    def fit_peptide_fdr(self):
        tt_gpsms = self.grouper.exclusive_match_groups['target_peptide_target_glycan']
        dt_gpsms = self.grouper.exclusive_match_groups['decoy_peptide_target_glycan']

        target_peptides = [SpectrumMatch(
            t.scan, t.target, t.score_set.peptide_score, True) for t in tt_gpsms]
        decoy_peptides = [SpectrumMatch(
            t.scan, t.target, t.score_set.peptide_score, True) for t in dt_gpsms]

        peptide_fdr = TargetDecoyAnalyzer(target_peptides, decoy_peptides)
        self.log("5%% Peptide FDR = %f (%d)" % (
            peptide_fdr.score_for_fdr(0.05),
            peptide_fdr.n_targets_above_threshold(peptide_fdr.score_for_fdr(0.05))))

        self.log("1%% Peptide FDR = %f (%d)" % (
            peptide_fdr.score_for_fdr(0.01),
            peptide_fdr.n_targets_above_threshold(peptide_fdr.score_for_fdr(0.01))))
        self.peptide_fdr = peptide_fdr
        return self.peptide_fdr

    def fit_glycopeptide_fdr(self):
        tt_gpsms = self.grouper.exclusive_match_groups['target_peptide_target_glycan']
        dd_gpsms = self.grouper.exclusive_match_groups['decoy_peptide_decoy_glycan']

        target_total_scores = np.array([t.score_set.glycopeptide_score for t in tt_gpsms])
        decoy_total_scores = np.array([t.score_set.glycopeptide_score for t in dd_gpsms])

        glycopeptide_fdr = FiniteMixtureModelFDREstimator(
            decoy_total_scores[decoy_total_scores > 1],
            target_total_scores[target_total_scores > 1])
        glycopeptide_fdr.log = noop
        glycopeptide_fdr.fit()

        glycopeptide_fdr_mapping = NearestValueLookUp(zip(glycopeptide_fdr.estimate_fdr(glycopeptide_fdr.target_scores),
                                                          glycopeptide_fdr.target_scores))
        self.log("5%% Glycopeptide FDR = %f (%d)" % (glycopeptide_fdr_mapping[0.05], (
            target_total_scores > glycopeptide_fdr_mapping[0.05]).sum()))
        self.log("1%% Glycopeptide FDR = %f (%d)" % (glycopeptide_fdr_mapping[0.01], (
            target_total_scores > glycopeptide_fdr_mapping[0.01]).sum()))
        self.glycopeptide_fdr = glycopeptide_fdr
        return self.glycopeptide_fdr

    def _assign_total_fdr(self, solution_sets):
        for ts in solution_sets:
            for t in ts:
                t.q_value_set.peptide_q_value = self.peptide_fdr.q_value_map[t.score_set.peptide_score]
                t.q_value_set.glycan_q_value = self.glycan_fdr.fdr_map[t.score_set.glycan_score]
                t.q_value_set.glycopeptide_q_value = self.glycopeptide_fdr.fdr_map[t.score_set.glycopeptide_score]
                total = (t.q_value_set.peptide_q_value + t.q_value_set.glycan_q_value -
                         t.q_value_set.glycopeptide_q_value)
                total = max(t.q_value_set.peptide_q_value, t.q_value_set.glycan_q_value, total)
                t.q_value = t.q_value_set.total_q_value = total
            ts.q_value = ts.best_solution().q_value
        return solution_sets

    def _assign_peptide_fdr(self, solution_sets):
        for ts in solution_sets:
            for t in ts:
                total = t.q_value_set.peptide_q_value = self.peptide_fdr.q_value_map[
                    t.score_set.peptide_score]
                t.q_value_set.glycan_q_value = 0
                t.q_value_set.glycopeptide_q_value = 0
                t.q_value = t.q_value_set.total_q_value = total
            ts.q_value = ts.best_solution().q_value
        return solution_sets

    def _assign_glycan_fdr(self, solution_sets):
        for ts in solution_sets:
            for t in ts:
                t.q_value_set.peptide_q_value = 0
                total = t.q_value_set.glycan_q_value = self.glycan_fdr.fdr_map[t.score_set.glycan_score]
                t.q_value_set.glycopeptide_q_value = 0
                t.q_value = t.q_value_set.total_q_value = total
            ts.q_value = ts.best_solution().q_value
        return solution_sets

    def fit_total_fdr(self):
        if self.strategy == GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture:
            self._assign_total_fdr(self.grouper.match_type_groups['target_peptide_target_glycan'])
        elif self.strategy == GlycopeptideFDREstimationStrategy.peptide_fdr:
            self._assign_peptide_fdr(self.grouper.match_type_groups['target_peptide_target_glycan'])
        elif self.strategy == GlycopeptideFDREstimationStrategy.glycan_fdr:
            self._assign_glycan_fdr(self.grouper.match_type_groups['target_peptide_target_glycan'])
        else:
            raise NotImplementedError(self.strategy)
        return self.grouper

    def run(self):
        if self.strategy in (GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture,
                             GlycopeptideFDREstimationStrategy.glycan_fdr):
            self.fit_glycan_fdr()
        if self.strategy in (GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture,
                             GlycopeptideFDREstimationStrategy.peptide_fdr):
            self.fit_peptide_fdr()
        if self.strategy == GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture:
            self.fit_glycopeptide_fdr()
        return self.fit_total_fdr()
