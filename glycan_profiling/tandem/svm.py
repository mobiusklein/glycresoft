# Inspired heavily by mokapot
# https://github.com/wfondrie/mokapot
# https://pubs.acs.org/doi/10.1021/acs.jproteome.0c01010


from typing import Any, Dict, List

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from glycan_profiling.task import LoggingMixin

from . import target_decoy
from .spectrum_match import SpectrumMatch, MultiScoreSpectrumMatch


class SVMModelBase(LoggingMixin):
    dataset: target_decoy.TargetDecoyAnalyzer
    proxy_dataset: target_decoy.TargetDecoyAnalyzer

    scaler: StandardScaler
    model: LinearSVC
    model_args: Dict[str, Any]

    max_iter: int
    train_fdr: float
    worse_than_score: bool
    trained: bool

    def __init__(self, train_fdr=0.01, max_iter=10, **model_args):
        self.scaler = StandardScaler()
        self.train_fdr = train_fdr
        self.max_iter = max_iter
        self.trained = False
        self.model_args = model_args.copy()
        self.model = LinearSVC(dual=False, **self.model_args)
        self.worse_than_score = False
        self.dataset = None
        self.proxy_dataset = None

    def pack(self):
        self.dataset = None
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
        self.proxy_dataset = state['proxy_dataset']

    def init_model(self):
        return LinearSVC(dual=False, **self.model_args)

    def prepare_model(self, model, features: np.ndarray, labels: List[bool]):
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_ = self.scaler.transform(X)
        return self.model.decision_function(X_)

    def _get_psms_labels(
        self, tda: target_decoy.TargetDecoyAnalyzer, fdr_threshold: float = None
    ):
        if fdr_threshold is None:
            fdr_threshold = self.train_fdr
        labels = []
        is_target = []
        psms = []
        q_value_map = tda.q_value_map

        for t in tda.targets:
            labels.append(int(q_value_map[tda.get_score(t)] <= fdr_threshold))
            psms.append(t)
            is_target.append(True)

        n_targets = sum(labels)

        for d in tda.decoys:
            labels.append(-1)
            psms.append(d)
            is_target.append(False)

        n_decoys = len(tda.decoys)

        self.log(
            f"Selected {n_targets} target spectra and {n_decoys} decoy spectra selected for training"
        )
        return psms, np.array(labels), np.array(is_target)

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
        self, psms: List[SpectrumMatch], scores: List[float], is_target: List[bool]
    ) -> target_decoy.NearestValueLookUp:

        targets, decoys = self._wrap_psms_with_score(psms, scores, is_target)
        tda = target_decoy.TargetDecoyAnalyzer(targets, decoys, decoy_pseudocount=0)
        q_value_map = tda.q_value_map
        q_values = np.array([q_value_map[s] for s in scores])

        updated_labels = (q_values <= self.train_fdr).astype(int)
        updated_labels[~np.asanyarray(is_target)] = -1
        return q_value_map, q_values, updated_labels, tda

    def fit(self, tda: target_decoy.TargetDecoyAnalyzer):
        self.dataset = tda
        psms, labels, is_target = self._get_psms_labels(tda)

        features = self.extract_features(psms)
        normalized_features = self.scaler.fit_transform(features)
        starting_labels = labels

        model = self.prepare_model(self.model, normalized_features, labels)
        count_passing = []
        for i in range(self.max_iter):
            observations = normalized_features[labels.astype(bool), :]
            target_labels_i = (labels[labels.astype(bool)] + 1) / 2
            model.fit(observations, target_labels_i)

            scores = self.model.decision_function(normalized_features)
            q_value_map, q_values, labels, self.proxy_dataset = self.scores_to_q_values(
                psms, scores, is_target
            )
            self.log(f"... Round {i}: {(labels == 1).sum()}")

        if (labels == 1).sum() < (starting_labels == 1).sum():
            self.log(
                f"Model performing worse than initial {(labels == 1).sum()} < {(starting_labels == 1).sum()}"
            )
            self.worse_than_score = True
        self.model = model
        self.trained = True

    @property
    def weights(self):
        return self.model.coef_

    def plot(self, ax=None):
        psms, labels, is_target = self._get_psms_labels(self.dataset)
        features = self.extract_features(psms)
        normalized_features = self.scaler.fit_transform(features)
        scores = self.model.decision_function(normalized_features)
        _q_value_map, _q_values, _labels, proxy_tda = self.scores_to_q_values(
            psms, scores, is_target
        )
        ax = proxy_tda.plot(ax=ax)
        lo, hi = ax.get_xlim()
        lo = self.proxy_dataset.thresholds[0] - 1
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


class PeptideScoreSVMModel(SVMModelBase):

    def extract_features(self, psms: List[MultiScoreSpectrumMatch]) -> np.ndarray:
        features = np.zeros((len(psms), 2))
        for i, psm in enumerate(psms):
            features[i, :] = (
                psm.score_set.peptide_score,
                psm.score_set.peptide_coverage,
            )
        return features
