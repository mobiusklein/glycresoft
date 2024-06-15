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
from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np

from glypy.structure.glycan_composition import GlycanComposition
from glypy.utils.enum import EnumValue

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from glycresoft.task import TaskBase

from glycresoft.structure.enums import GlycopeptideFDREstimationStrategy
from glycresoft.tandem.spectrum_match.spectrum_match import FDRSet, MultiScoreSpectrumMatch, SpectrumMatch
from glycresoft.tandem.spectrum_match.solution_set import MultiScoreSpectrumSolutionSet, SpectrumSolutionSet
from glycresoft.tandem.target_decoy import (
    NearestValueLookUp,
    PeptideScoreTargetDecoyAnalyzer,
    FDREstimatorBase,
    PeptideScoreSVMModel
)
from glycresoft.tandem.glycopeptide.core_search import approximate_internal_size_of_glycan

from .mixture import (
    GammaMixture, GaussianMixture, GaussianMixtureWithPriorComponent,
    MixtureBase, TruncatedGaussianMixture, TruncatedGaussianMixtureWithPriorComponent)

from .journal import SolutionSetGrouper


def noop(*args, **kwargs):
    pass


def interpolate_from_zero(nearest_value_map: NearestValueLookUp, zero_value: float=1.0) -> NearestValueLookUp:
    smallest = nearest_value_map.items[0]
    X = np.linspace(0, smallest[0])
    Y = np.interp(X, [0, smallest[0]], [zero_value, smallest[1]])
    pairs = list(zip(X, Y))
    pairs.extend(nearest_value_map.items)
    return nearest_value_map.__class__(pairs)


class FiniteMixtureModelFDREstimatorBase(FDREstimatorBase):
    decoy_scores: np.ndarray
    target_scores: np.ndarray
    decoy_mixture: Optional[MixtureBase]
    target_mixture: Optional[MixtureBase]
    fdr_map: Optional[NearestValueLookUp]

    def __init__(self, decoy_scores, target_scores):
        self.decoy_scores = np.array(decoy_scores)
        self.target_scores = np.array(target_scores)
        self.decoy_mixture = None
        self.target_mixture = None
        self.fdr_map = None

    def estimate_posterior_error_probability(self, X: np.ndarray) -> np.ndarray:
        return self.target_mixture.prior.score(X) * self.pi0 / self.target_mixture.score(X)

    def get_count_for_fdr(self, q_value: float):
        target_scores = self.target_scores
        target_scores = np.sort(target_scores)
        fdr = np.sort(self.estimate_fdr())[::-1]
        if len(fdr) == 0:
            return 0, 0
        ii = np.where(fdr < q_value)[0]
        if len(ii) == 0:
            return float('inf'), 0
        i = ii[0]
        score_for_fdr = target_scores[i]
        target_counts = (target_scores >= score_for_fdr).sum()
        return score_for_fdr, target_counts

    @property
    def pi0(self):
        return self.target_mixture.weights[-1]

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

        dmin = 0
        dmax = 1
        tmin = 0
        tmax = 1
        if len(self.target_scores):
            tmin = self.target_scores.min()
            tmax = self.target_scores.max()
        if len(self.decoy_scores):
            dmin = self.decoy_scores.min()
            dmax = self.decoy_scores.max()
        points = np.linspace(
            min(tmin, dmin),
            max(tmax, dmax),
            10000)
        target_scores = np.sort(self.target_scores)
        target_counts = [(self.target_scores >= i).sum() for i in points]
        decoy_counts = [(self.decoy_scores >= i).sum() for i in points]
        fdr = self.estimate_fdr(target_scores)
        at_5_percent = np.where(fdr < 0.05)[0][0]
        at_1_percent = np.where(fdr < 0.01)[0][0]
        line1 = ax.plot(points, target_counts,
                        label='Target', color='steelblue')
        line2 = ax.plot(points, decoy_counts, label='Decoy', color='coral')
        line4 = ax.vlines(target_scores[at_5_percent], 0, np.max(
            target_counts), linestyle='--', color='green', lw=0.75, label='5% FDR')
        line5 = ax.vlines(target_scores[at_1_percent], 0, np.max(
            target_counts), linestyle='--', color='skyblue', lw=0.75, label='1% FDR')
        ax.set_ylabel("# Matches Retained")
        ax.set_xlabel("Score")
        ax2 = ax.twinx()
        ax2.set_ylabel("FDR")
        line3 = ax2.plot(target_scores, fdr, label='FDR',
                         color='grey', linestyle='--')
        ax.legend(
            [line1[0], line2[0], line3[0], line4, line5],
            ['Target', 'Decoy', 'FDR', '5% FDR', '1% FDR'], frameon=False)

        lo, hi = ax.get_ylim()
        lo = max(lo, 0)
        ax.set_ylim(lo, hi)
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(0, hi)

        lo, hi = ax.get_xlim()
        ax.set_xlim(-1, hi)
        lo, hi = ax2.get_xlim()
        ax2.set_xlim(-1, hi)
        return ax

    def fit(self, max_components: int=10,
            max_target_components: Optional[int]=None,
            max_decoy_components: Optional[int]=None) -> NearestValueLookUp:
        if not max_target_components:
            max_target_components = max_components
        if not max_decoy_components:
            max_decoy_components = max_components
        self.estimate_decoy_distributions(max_decoy_components)
        self.estimate_target_distributions(max_target_components)
        fdr = self.estimate_fdr(self.target_scores)
        self.fdr_map = NearestValueLookUp(zip(self.target_scores, fdr))
        # Since 0 is not in the domain of the model, we need to include it by interpolating from 1 to the smallest
        # fitted value.
        if self.fdr_map.items[0][0] > 0:
            self.fdr_map = interpolate_from_zero(self.fdr_map)
        return self.fdr_map

    def estimate_fdr(self, X: np.ndarray=None) -> np.ndarray:
        if X is None:
            X = self.target_scores
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

    def _select_best_number_of_components(self, max_components: int,
                                          fitter: Type[MixtureBase],
                                          scores: np.ndarray, **kwargs) -> MixtureBase:
        models = []
        bics = []
        for i in range(1, max_components + 1):
            self.debug("Fitting %d Components" % (i,))
            iteration_kwargs = kwargs.copy()
            iteration_kwargs['n_components'] = i
            model = fitter(scores, **iteration_kwargs)
            bic = model.bic(scores)
            models.append(model)
            bics.append(bic)
            self.debug("BIC: %g" % (bic,))
        i = np.argmin(bics)
        self.debug("Selected %d Components" % (i + 1,))
        return models[i]

    def estimate_decoy_distributions(self, max_components: int=10):
        raise NotImplementedError()

    def estimate_target_distributions(self, max_components: int=10) -> MixtureBase:
        n = len(self.target_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few target observations")
            self.target_mixture = GaussianMixtureWithPriorComponent(
                [1.0], [1.0], self.decoy_mixture, [0.5, 0.5])
            return self.target_mixture
        self.target_mixture = self._select_best_number_of_components(
            max_components,
            GaussianMixtureWithPriorComponent.fit,
            self.target_scores,
            prior=self.decoy_mixture,
            deterministic=True)
        return self.target_mixture

    def score(self, spectrum_match: Union[SpectrumMatch, float], assign: bool = False) -> float:
        return self.fdr_map[
            spectrum_match.score
            if isinstance(spectrum_match, SpectrumMatch)
            else spectrum_match]


class FiniteMixtureModelFDREstimatorSeparated(FiniteMixtureModelFDREstimatorBase):
    decoy_model_type = GammaMixture
    target_model_type = GaussianMixture

    def estimate_decoy_distributions(self, max_components: int = 10):
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.decoy_mixture = self.decoy_model_type(
                [1.0], [1.0], [1.0])
            return self.decoy_mixture

        self.decoy_mixture = self._select_best_number_of_components(
            max_components, self.decoy_model_type.fit, self.decoy_scores)
        return self.decoy_mixture

    def estimate_target_distributions(self, max_components: int = 10):
        n = len(self.target_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few target observations")
            self.target_mixture = self.target_model_type(
                [1.0], [1.0], [1.0])
            return self.target_mixture
        self.target_mixture = self._select_best_number_of_components(
            max_components,
            GaussianMixture.fit,
            self.target_scores,
            deterministic=True)
        return self.target_mixture

    def estimate_posterior_error_probability(self, X: np.ndarray) -> np.ndarray:
        d = self.decoy_mixture.score(X)
        t = self.target_mixture.score(X)
        pi0 = self.pi0
        return (pi0 * d) / (pi0 * d + (1 - pi0) * t)

    def get_count_for_fdr(self, q_value):
        target_scores = self.target_scores
        target_scores = np.sort(target_scores)
        fdr = np.sort(self.estimate_fdr())[::-1]

        i = np.where(fdr < q_value)[0][0]
        score_for_fdr = target_scores[i]
        target_counts = (target_scores >= score_for_fdr).sum()
        return score_for_fdr, target_counts

    @property
    def pi0(self):
        pi0 = len(self.decoy_scores) / len(self.target_scores)
        return pi0


class FiniteMixtureModelFDREstimatorDecoyGamma(FiniteMixtureModelFDREstimatorBase):

    def estimate_decoy_distributions(self, max_components=10):
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.decoy_mixture = GammaMixture([1.0], [1.0], [1.0])
            return self.decoy_mixture

        self.decoy_mixture = self._select_best_number_of_components(
            max_components, GammaMixture.fit, self.decoy_scores)
        return self.decoy_mixture


class FiniteMixtureModelFDREstimatorDecoyGaussian(FiniteMixtureModelFDREstimatorBase):

    def estimate_decoy_distributions(self, max_components=10):
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.decoy_mixture = GaussianMixture([1.0], [1.0], [1.0])
            return self.decoy_mixture

        self.decoy_mixture = self._select_best_number_of_components(
            max_components, GaussianMixture.fit, self.decoy_scores)
        return self.decoy_mixture


FiniteMixtureModelFDREstimator = FiniteMixtureModelFDREstimatorDecoyGamma


class FiniteMixtureModelFDREstimatorHalfGaussian(FiniteMixtureModelFDREstimatorBase):

    def estimate_decoy_distributions(self, max_components=10):
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.decoy_mixture = TruncatedGaussianMixture([0], [1.0], [1.0])
            return self.decoy_mixture

        self.decoy_mixture = self._select_best_number_of_components(
            max_components, TruncatedGaussianMixture.fit, self.decoy_scores)
        return self.decoy_mixture

    def estimate_target_distributions(self, max_components=10):
        n = len(self.target_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few target observations")
            self.target_mixture = TruncatedGaussianMixtureWithPriorComponent(
                [0.0], [1.0], self.decoy_mixture, [0.5, 0.5])
            return self.target_mixture
        self.target_mixture = self._select_best_number_of_components(
            max_components,
            TruncatedGaussianMixtureWithPriorComponent.fit,
            self.target_scores,
            prior=self.decoy_mixture,
            deterministic=True)
        return self.target_mixture


class MultivariateMixtureModel(FDREstimatorBase):
    models: List[FiniteMixtureModelFDREstimatorBase]

    def __init__(self, models):
        self.models = models

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, i):
        return self.models[i]

    def __len__(self):
        return len(self.models)

    def is_combined(self, mixture_model):
        tm = mixture_model.target_mixture
        if hasattr(tm, 'prior'):
            return True
        return False

    def _compute_per_feature_probabilities(self, X: List[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if X is None:
            X = [m.target_scores for m in self.models]

        decoy_series = []
        target_series = []
        for i, (model, x) in enumerate(zip(self.models, X)):
            if self.is_combined(model):
                decoy_p = model.decoy_mixture.score(x)
                w_decoy = model.pi0
                target_decoy_p = model.target_mixture.score(x)
                target_p = (target_decoy_p - w_decoy * decoy_p)
                decoy_p *= w_decoy
            else:
                decoy_p = model.decoy_mixture.score(x)
                target_p = model.target_mixture.score(x)
            decoy_series.append(decoy_p)
            target_series.append(target_p)
        return decoy_series, target_series

    def estimate_pi0(self, X: List[np.ndarray]=None) -> Tuple[float, np.ndarray, np.ndarray]:
        decoy_series, target_series = self._compute_per_feature_probabilities(X)
        decoy = np.prod(decoy_series, axis=0)
        target = np.prod(target_series, axis=0)
        return (decoy / (decoy + target)).mean(), decoy, target

    def _compute_overlap_probabilities(self, decoy_series: List[np.ndarray],
                                       target_series: List[np.ndarray],
                                       pi0s: List[float]) -> List[np.ndarray]:
        mixed_terms = []
        for i, d in enumerate(decoy_series):
            for j, t in enumerate(target_series):
                if i != j:
                    mixed_terms.append(d * t / (1 - pi0s[j]) * pi0s[j])
        return mixed_terms

    def estimate_posterior_error_probability(self, X: List[np.ndarray] = None, correlated: bool=False) -> np.ndarray:
        if X is None:
            X = [m.target_scores for m in self.models]

        decoy_series, target_series = self._compute_per_feature_probabilities(X)
        if correlated:
            # mixed_terms = self._compute_overlap_probabilities(
            #     decoy_series, target_series, [m.pi0 for m in self])
            raise NotImplementedError("Correlated features are not implemented")

        decoy = np.prod(decoy_series, axis=0)
        target = np.prod(target_series, axis=0)
        if self.is_combined(self.models[0]):
            pass
        else:
            pi_0 = self.models[0].pi0
            decoy *= pi_0
            target *= (1 - pi_0)
        if not correlated:
            return decoy / (target + decoy)
        raise NotImplementedError("Correlated features are not implemented")

    def estimate_fdr(self, X: List[np.ndarray] = None, correlated: bool=False):
        if X is None:
            X = [m.target_scores for m in self.models]
        X = np.stack(X, axis=-1)
        X_ = X[np.lexsort(X[:, ::-1].T)[::-1], :]
        pep = self.estimate_posterior_error_probability(list(X_.T))
        fdr = np.cumsum(pep) / np.arange(1, len(X_) + 1)
        fdr[np.isnan(fdr)] = 1.0
        fdr_descending = fdr[::-1]
        for i in range(1, fdr_descending.shape[0]):
            if fdr_descending[i - 1] < fdr_descending[i]:
                fdr_descending[i] = fdr_descending[i - 1]
        fdr = fdr_descending[::-1]
        fdr = fdr[::-1][np.searchsorted(X_[::-1, 0], X[:, 0])]
        return fdr

    def fit(self, correlated: bool = False, *args, **kwargs) -> NearestValueLookUp:
        fdr = self.estimate_fdr(correlated=correlated)
        self.fdr_map = NearestValueLookUp(
            zip(self.models[0].target_scores, fdr))
        # Since 0 is not in the domain of the model, we need to include it by
        # interpolating from 1 to the smallest fitted value.
        if self.fdr_map.items[0][0] > 0:
            self.fdr_map = interpolate_from_zero(self.fdr_map)
        return self.fdr_map

    def get_count_for_fdr(self, q_value):
        target_scores = self.models[0].target_scores
        target_scores = np.sort(target_scores)
        fdr = np.sort(self.estimate_fdr())[::-1]

        i = np.where(fdr < q_value)[0][0]
        score_for_fdr = target_scores[i]
        target_counts = (target_scores >= score_for_fdr).sum()
        return score_for_fdr, target_counts

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        target_scores = self.models[0].target_scores
        decoy_scores = self.models[0].decoy_scores

        dmin = 0
        dmax = float('inf')
        tmin = 0
        tmax = float('inf')
        if len(target_scores):
            tmin = target_scores.min()
            tmax = target_scores.max()
        if len(decoy_scores):
            dmin = decoy_scores.min()
            dmax = decoy_scores.max()
        points = np.linspace(
            min(tmin, dmin),
            max(tmax, dmax),
            10000)

        target_scores = np.sort(target_scores)
        target_counts = [(target_scores >= i).sum() for i in points]
        decoy_counts = [(decoy_scores >= i).sum() for i in points]

        fdr = np.sort(self.estimate_fdr())[::-1]

        at_5_percent = np.where(fdr < 0.05)[0][0]
        at_1_percent = np.where(fdr < 0.01)[0][0]

        line1 = ax.plot(points, target_counts,
                        label='Target', color='steelblue')
        line2 = ax.plot(points, decoy_counts, label='Decoy', color='coral')
        line4 = ax.vlines(target_scores[at_5_percent], 0, np.max(
            target_counts), linestyle='--', color='green', lw=0.75, label='5% FDR')
        line5 = ax.vlines(target_scores[at_1_percent], 0, np.max(
            target_counts), linestyle='--', color='skyblue', lw=0.75, label='1% FDR')
        ax.set_ylabel("# Matches Retained")
        ax.set_xlabel("Score")
        ax2 = ax.twinx()
        ax2.set_ylabel("FDR")
        line3 = ax2.plot(target_scores, fdr, label='FDR',
                         color='grey', linestyle='--')
        ax.legend(
            [line1[0], line2[0], line3[0], line4, line5],
            ['Target', 'Decoy', 'FDR', '5% FDR', '1% FDR'], frameon=False)

        lo, hi = ax.get_ylim()
        lo = max(lo, 0)
        ax.set_ylim(lo, hi)
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(0, hi)

        lo, hi = ax.get_xlim()
        ax.set_xlim(-1, hi)
        lo, hi = ax2.get_xlim()
        ax2.set_xlim(-1, hi)
        return ax


class GlycanSizeCalculator(object):
    glycan_size_cache: Dict[str, int]

    def __init__(self):
        self.glycan_size_cache = dict()

    def get_internal_size(self, glycan_composition: GlycanComposition) -> int:
        key = str(glycan_composition)
        if key in self.glycan_size_cache:
            return self.glycan_size_cache[key]
        n = approximate_internal_size_of_glycan(glycan_composition)
        self.glycan_size_cache[key] = n
        return n

    def __call__(self, glycan_composition: GlycanComposition) -> int:
        return self.get_internal_size(glycan_composition)


class GlycopeptideFDREstimator(TaskBase):
    display_fields = False
    minimum_glycan_size: int = 3
    minimum_score: float = 1

    strategy: EnumValue
    grouper: 'SolutionSetGrouper'

    glycan_fdr: FiniteMixtureModelFDREstimator
    peptide_fdr: Union[PeptideScoreTargetDecoyAnalyzer, PeptideScoreSVMModel]
    glycopeptide_fdr: FiniteMixtureModelFDREstimator

    _peptide_fdr_estimator_type: Type[PeptideScoreTargetDecoyAnalyzer]

    def __init__(self, groups: 'SolutionSetGrouper',
                 strategy: Optional[GlycopeptideFDREstimationStrategy] = None,
                 peptide_fdr_estimator: Optional[Type[PeptideScoreTargetDecoyAnalyzer]] = None):
        if peptide_fdr_estimator is None:
            peptide_fdr_estimator = PeptideScoreTargetDecoyAnalyzer
        if strategy is None:
            strategy = GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture
        else:
            strategy = GlycopeptideFDREstimationStrategy[strategy]
        self.strategy = strategy
        self.grouper = groups
        self.glycan_fdr = None
        self.peptide_fdr = None
        self.glycopeptide_fdr = None
        self._peptide_fdr_estimator_type = peptide_fdr_estimator

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
            decoy_glycan_scores[(size_mask > self.minimum_glycan_size) & (
                decoy_glycan_scores > self.minimum_score)],
            target_scores=target_glycan_scores[target_glycan_scores > self.minimum_score])

        glycan_fdr.fit()
        glycan_fdr.summarize("Glycan FDR")

        self.glycan_fdr = glycan_fdr
        return self.glycan_fdr

    def fit_peptide_fdr(self):
        tt_gpsms = self.grouper.exclusive_match_groups['target_peptide_target_glycan']
        dt_gpsms = self.grouper.exclusive_match_groups['decoy_peptide_target_glycan']

        target_peptides = tt_gpsms
        decoy_peptides = dt_gpsms

        peptide_fdr = self._peptide_fdr_estimator_type(target_peptides, decoy_peptides)
        peptide_fdr.summarize("Peptide FDR")

        self.peptide_fdr = peptide_fdr
        return self.peptide_fdr

    def fit_glycopeptide_fdr(self):
        tt_gpsms = self.grouper.exclusive_match_groups['target_peptide_target_glycan']
        dd_gpsms = self.grouper.exclusive_match_groups['decoy_peptide_decoy_glycan']

        target_total_scores = np.array([t.score_set.glycopeptide_score for t in tt_gpsms])
        decoy_total_scores = np.array([t.score_set.glycopeptide_score for t in dd_gpsms])

        glycopeptide_fdr = FiniteMixtureModelFDREstimator(
            decoy_total_scores[decoy_total_scores > self.minimum_score],
            target_total_scores[target_total_scores > self.minimum_score])

        glycopeptide_fdr.fit()
        glycopeptide_fdr.summarize("Glycopeptide FDR")

        self.glycopeptide_fdr = glycopeptide_fdr
        return self.glycopeptide_fdr

    def _assign_joint_fdr(self, solution_sets: List[MultiScoreSpectrumSolutionSet]):
        for ts in solution_sets:
            for t in ts:
                t.q_value_set.peptide_q_value = self.peptide_fdr.score(t, assign=False)
                t.q_value_set.glycan_q_value = self.glycan_fdr.fdr_map[t.score_set.glycan_score]
                t.q_value_set.glycopeptide_q_value = self.glycopeptide_fdr.fdr_map[t.score_set.glycopeptide_score]
                total = (t.q_value_set.peptide_q_value + t.q_value_set.glycan_q_value -
                         t.q_value_set.glycopeptide_q_value)
                total = max(t.q_value_set.peptide_q_value, t.q_value_set.glycan_q_value, total)
                t.q_value = t.q_value_set.total_q_value = total
            if isinstance(ts, SpectrumSolutionSet):
                ts.q_value = ts.best_solution().q_value
        return solution_sets

    def _assign_minimum_fdr(self, solution_sets: List[MultiScoreSpectrumSolutionSet]):
        for ts in solution_sets:
            for t in ts:
                t.q_value_set.peptide_q_value = self.peptide_fdr.score(t, assign=False)
                t.q_value_set.glycan_q_value = self.glycan_fdr.fdr_map[t.score_set.glycan_score]
                t.q_value_set.glycopeptide_q_value = self.glycopeptide_fdr.fdr_map[
                    t.score_set.glycopeptide_score]
                total = min(t.q_value_set.peptide_q_value, t.q_value_set.glycan_q_value)
                t.q_value = t.q_value_set.total_q_value = total
            ts.q_value = ts.best_solution().q_value
        return solution_sets

    def _assign_peptide_fdr(self, solution_sets: List[MultiScoreSpectrumSolutionSet]):
        for ts in solution_sets:
            for t in ts:
                total = t.q_value_set.peptide_q_value = self.peptide_fdr.score(t, assign=False)
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

    def fit_total_fdr(self, solution_sets: Optional[List[MultiScoreSpectrumSolutionSet]] = None):
        if solution_sets is None:
            solution_sets = self.grouper.match_type_groups['target_peptide_target_glycan']
        if self.strategy == GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture:
            self._assign_joint_fdr(solution_sets)
        elif self.strategy == GlycopeptideFDREstimationStrategy.peptide_fdr:
            self._assign_peptide_fdr(solution_sets)
        elif self.strategy == GlycopeptideFDREstimationStrategy.glycan_fdr:
            self._assign_glycan_fdr(solution_sets)
        elif self.strategy == GlycopeptideFDREstimationStrategy.peptide_or_glycan:
            self._assign_minimum_fdr(solution_sets)
        else:
            raise NotImplementedError(self.strategy)
        return self.grouper

    def score_all(self, solution_set: MultiScoreSpectrumSolutionSet):
        self.fit_total_fdr([solution_set])
        return solution_set

    def score(self, gpsm: MultiScoreSpectrumMatch, assign: bool=False):
        q_value = gpsm.q_value
        q_value_set = gpsm.q_value_set
        placeholder = FDRSet.default()
        gpsm.q_value_set = placeholder
        self.score_all(MultiScoreSpectrumSolutionSet(gpsm.scan, [gpsm]))
        result_q_value_set = placeholder
        if not assign:
            gpsm.q_value = q_value
            gpsm.q_value_set = q_value_set
        return result_q_value_set

    def run(self):
        if self.strategy in (GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture,
                             GlycopeptideFDREstimationStrategy.glycan_fdr,
                             GlycopeptideFDREstimationStrategy.peptide_or_glycan):
            self.fit_glycan_fdr()
        if self.strategy in (GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture,
                             GlycopeptideFDREstimationStrategy.peptide_fdr,
                             GlycopeptideFDREstimationStrategy.peptide_or_glycan):
            self.fit_peptide_fdr()
        if self.strategy in (GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture,
                             GlycopeptideFDREstimationStrategy.peptide_or_glycan):
            self.fit_glycopeptide_fdr()
        self.fit_total_fdr()
        return self.grouper

    def pack(self):
        self.grouper = SolutionSetGrouper([])
        if self.peptide_fdr is not None:
            self.peptide_fdr.pack()
