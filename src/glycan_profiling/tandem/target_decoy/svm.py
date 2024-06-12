# Inspired heavily by mokapot
# https://github.com/wfondrie/mokapot
# https://pubs.acs.org/doi/10.1021/acs.jproteome.0c01010


from typing import Any, Dict, List, Optional, Tuple

from array import ArrayType as Array

import numpy as np
from numpy.typing import NDArray

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from ..spectrum_match import SpectrumMatch, MultiScoreSpectrumMatch

from .base import TargetDecoyAnalyzer, NearestValueLookUp, PeptideScoreTargetDecoyAnalyzer, FDREstimatorBase


def _transform_fast(self: StandardScaler, X: np.ndarray) -> np.ndarray:
    Y = (X - self.mean_)
    Y /= self.scale_
    return  Y


def _fast_decision_function(self: LinearSVC, X: np.ndarray) -> np.ndarray:
    Y = X.dot(self.coef_[0, :])
    Y += self.intercept_
    return Y


class SVMModelBase(FDREstimatorBase):
    dataset: TargetDecoyAnalyzer
    proxy_dataset: TargetDecoyAnalyzer

    scaler: StandardScaler
    model: LinearSVC
    model_args: Dict[str, Any]

    max_iter: int
    train_fdr: float
    worse_than_score: bool
    trained: bool

    def __init__(self, target_matches: Optional[List[MultiScoreSpectrumMatch]]=None,
                 decoy_matches: Optional[List[MultiScoreSpectrumMatch]]=None,
                 train_fdr: float=0.01, max_iter: int=10, **model_args):
        self.scaler = StandardScaler()
        self.train_fdr = train_fdr
        self.max_iter = max_iter
        self.trained = False
        self.model_args = model_args.copy()
        self.model = LinearSVC(dual=False, **self.model_args)
        self.worse_than_score = False
        self.dataset = None
        self.proxy_dataset = None

        if target_matches is not None and decoy_matches is not None:
            tda = self._wrap_dataset(target_matches, decoy_matches)
            _, count = tda.get_count_for_fdr(self.train_fdr)
            if count < 100:
                self.warn(f"Found {count} targets passing {self.train_fdr} FDR threshold, "
                          "too few observations to fit a reliable model")
            self.fit(tda)

    def feature_names(self) ->  List[str]:
        raise NotImplementedError()

    def _wrap_dataset(self, target_matches: List[MultiScoreSpectrumMatch],
                      decoy_matches: List[MultiScoreSpectrumMatch]):
        tda = TargetDecoyAnalyzer(target_matches, decoy_matches, decoy_pseudocount=0.0)
        return tda

    def pack(self):
        self.dataset.pack()
        self.proxy_dataset.pack()

    def __getstate__(self):
        return {
            "scaler": self.scaler,
            "model": self.model,
            "model_args": self.model_args,
            "trained": self.trained,
            "worse_than_score": self.worse_than_score,
            "max_iter": self.max_iter,
            "train_fdr": self.train_fdr,
            "dataset": self.dataset,
            "proxy_dataset": self.proxy_dataset
        }

    def __setstate__(self, state):
        self.scaler = state['scaler']
        self.model = state['model']
        self.model_args = state['model_args']
        self.trained = state['trained']
        self.worse_than_score = state['worse_than_score']
        self.max_iter = state['max_iter']
        self.train_fdr = state['train_fdr']
        self.dataset = state['dataset']
        self.proxy_dataset = state['proxy_dataset']

    def init_model(self):
        return LinearSVC(dual=False, **self.model_args)

    def prepare_model(self, model, features: np.ndarray, labels: List[bool]):
        return model

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        X_ = _transform_fast(self.scaler, X)
        return _fast_decision_function(self.model, X_)

    def _get_psms_labels(
        self, tda: TargetDecoyAnalyzer, fdr_threshold: float = None
    ) -> Tuple[List[MultiScoreSpectrumMatch], NDArray[np.bool_], NDArray[np.bool_]]:
        if fdr_threshold is None:
            fdr_threshold = self.train_fdr
        labels = Array('b')
        is_target = Array('b')
        psms = []
        q_value_map = tda.q_value_map

        for t in tda.targets:
            labels.append(int(q_value_map[tda.get_score(t)] <= fdr_threshold))
            psms.append(t)
            is_target.append(True)

        n_targets = np.frombuffer(labels, dtype=bool).sum()

        for d in tda.decoys:
            labels.append(-1)
            psms.append(d)
            is_target.append(False)

        n_decoys = len(tda.decoys)

        self.log(
            f"Selected {n_targets} target spectra and {n_decoys} decoy spectra selected for training"
        )
        return psms, np.frombuffer(labels, dtype=np.int8), np.frombuffer(is_target, dtype=bool)

    def extract_features(self, psms: List[MultiScoreSpectrumMatch]) -> np.ndarray:
        '''Override in subclass to construct feature matrix for SVM model'''
        raise NotImplementedError()

    def _wrap_psms_with_score(
        self, psms: List[SpectrumMatch], scores: List[float], is_target: List[bool]
    ) -> List[SpectrumMatch]:
        targets = []
        decoys = []
        for psm, score, is_t in zip(psms, scores, is_target):
            wrapped = SpectrumMatch(psm.scan, psm.target, score)
            if is_t:
                targets.append(wrapped)
            else:
                decoys.append(wrapped)
        return targets, decoys

    def scores_to_q_values(
        self, psms: List[SpectrumMatch], scores: np.ndarray, is_target: List[bool]
    ) -> Tuple[NearestValueLookUp, np.ndarray, ]:

        targets, decoys = self._wrap_psms_with_score(psms, scores, is_target)
        tda = TargetDecoyAnalyzer(targets, decoys, decoy_pseudocount=0)
        q_value_map = tda.q_value_map
        q_values = np.array([q_value_map[s] for s in scores])

        updated_labels = (q_values <= self.train_fdr).astype(int)
        updated_labels[~np.asanyarray(is_target)] = -1
        return q_value_map, q_values, updated_labels, tda

    def fit(self, tda: TargetDecoyAnalyzer):
        self.dataset = tda
        psms, labels, is_target = self._get_psms_labels(tda, self.train_fdr)

        k = is_target.sum()
        has_no_decoys = k == len(is_target)
        has_no_targets = k == 0
        if has_no_decoys or has_no_targets:
            self.worse_than_score = True
            self.proxy_dataset = self.dataset
            return

        features = self.extract_features(psms)
        normalized_features = self.scaler.fit_transform(features)
        starting_labels = labels

        model = self.prepare_model(self.model, normalized_features, labels)
        for i in range(self.max_iter):
            observations = normalized_features[labels.astype(bool), :]
            target_labels_i = (labels[labels.astype(bool)] + 1) / 2
            if (target_labels_i == 1).sum() == 0 or (target_labels_i == 0).sum() == 0:
                self.warn("Found only one observation class, cannot proceed with model fitting")
                break
            model.fit(observations, target_labels_i)

            scores = self.model.decision_function(normalized_features)
            _q_value_map, _q_values, labels, self.proxy_dataset = self.scores_to_q_values(
                psms, scores, is_target
            )
            self.log(f"... Round {i}: {(labels == 1).sum()}")

        if (labels == 1).sum() < (starting_labels == 1).sum():
            self.log(
                f"Model performing worse than initial {(labels == 1).sum()} < {(starting_labels == 1).sum()}"
            )
            self.worse_than_score = True
            assert self.dataset is not None
        self.model = model
        self.trained = True
        self._normalized_features = normalized_features

    @property
    def weights(self):
        try:
            return self.model.coef_
        except AttributeError:
            self.log("Model coefficients not yet fit")
            return [[]]

    def plot(self, ax=None):
        if self.worse_than_score:
            return self.dataset.plot(ax=ax)
        ax = self.proxy_dataset.plot(ax=ax)
        lo, hi = ax.get_xlim()
        lo = self.proxy_dataset.thresholds[0] - 0.25
        ax.set_xlim(lo, hi)
        ax.set_xlabel("SVM Score")
        return ax

    def get_count_for_fdr(self, q_value: float):
        return self.proxy_dataset.get_count_for_fdr(q_value)

    @property
    def q_value_map(self):
        return self.proxy_dataset.q_value_map

    @property
    def fdr_map(self):
        return self.q_value_map

    def score(self, spectrum_match: MultiScoreSpectrumMatch, assign=None):
        if assign:
            self.warn("The assign argument is a no-op")
        if self.worse_than_score:
            return self.dataset.score(spectrum_match, assign=False)
        x = self.extract_features([spectrum_match])
        y = self.predict(x)
        return self.q_value_map[y[0]]

    def summarize(self, name: Optional[str]=None):
        if name is None:
            name = "FDR"
        if self.worse_than_score:
            self.dataset.summarize(name)
            return

        self.log("Feature Weights:")
        feature_names = self.feature_names() + ['intercept']
        feature_values = list(self.weights[0]) + [self.model.intercept_]
        for fname, fval in zip(feature_names, feature_values):
            self.log(f"... {fname}: {fval}")

        threshold_05, count_05 = self.get_count_for_fdr(0.05)
        self.log(f"5% {name} = {threshold_05:0.3f} ({count_05})")
        threshold_01, count_01 = self.get_count_for_fdr(0.01)
        self.log(f"1% {name} = {threshold_01:0.3f} ({count_01})")


class PeptideScoreSVMModel(SVMModelBase):

    def _wrap_dataset(self, target_matches: List[MultiScoreSpectrumMatch],
                      decoy_matches: List[MultiScoreSpectrumMatch]):
        tda = PeptideScoreTargetDecoyAnalyzer(
            target_matches, decoy_matches, decoy_pseudocount=0.0)
        return tda

    def feature_names(self) -> List[str]:
        return [
            "peptide_score",
            "peptide_coverage",
        ]

    def extract_features(self, psms: List[MultiScoreSpectrumMatch]) -> np.ndarray:
        features = np.zeros((len(psms), 2))
        for i, psm in enumerate(psms):
            features[i, :] = (
                psm.score_set.peptide_score,
                psm.score_set.peptide_coverage,
            )
        return features
