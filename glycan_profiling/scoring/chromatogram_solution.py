from collections import namedtuple, OrderedDict
from operator import mul
try:
    reduce
except NameError:
    from functools import reduce

import numpy as np

from .shape_fitter import AdaptiveMultimodalChromatogramShapeFitter
from .spacing_fitter import ChromatogramSpacingFitter
from .charge_state import UniformChargeStateScoringModel
from .isotopic_fit import IsotopicPatternConsistencyFitter
from .base import symbolic_composition

from glycan_profiling import symbolic_expression
from glycan_profiling.chromatogram_tree import ChromatogramInterface
from glycopeptidepy import HashableGlycanComposition


epsilon = 1e-6

scores = namedtuple("scores", ["line_score", "isotopic_fit", "spacing_fit", "charge_count"])
scores_adducted = namedtuple(
    "scores_adducted",
    ["line_score", "isotopic_fit", "spacing_fit", "charge_count",
     "adduct_score"])


def logit(x):
    return np.log(x) - np.log(1 - x)


def logitsum(xs):
    total = 0
    for x in xs:
        total += logit(x)
    return total


def prod(*x):
    return reduce(mul, x, 1)


class ScorerBase(object):
    def __init__(self, *args, **kwargs):
        self.feature_set = OrderedDict()

    def add_feature(self, scoring_feature):
        if scoring_feature is None:
            return
        feature_type = scoring_feature.get_feature_type()
        self.feature_set[feature_type] = scoring_feature

    def features(self):
        for feature in self.feature_set.values():
            yield feature

    def compute_scores(self, chromatogram):
        raise NotImplementedError()

    def logitscore(self, chromatogram):
        score = logitsum(self.compute_scores(chromatogram))
        return score

    def score(self, chromatogram):
        score = reduce(mul, self.compute_scores(chromatogram), 1.0)
        return score


class ChromatogramScoreSet(object):
    def __init__(self, scores):
        self.scores = OrderedDict(scores)

    def __iter__(self):
        return iter(self.scores.values())

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.scores)[1:-1])

    def __getattr__(self, key):
        if key == "scores":
            raise AttributeError(key)
        return self.scores[key]

    def items(self):
        return self.scores.items()


class DummyScorer(ScorerBase):
    def __init__(*args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        return 1.0

    @property
    def line_test(self):
        return 0.0

    @property
    def spacing_fit(self):
        return 0.0

    @property
    def mean_fit(self):
        return 0.0

    def compute_scores(self, chromatogram):
        line_score = 1 - epsilon
        isotopic_fit = 1 - epsilon
        spacing_fit = 1 - epsilon
        charge_count = 1 - epsilon
        return ChromatogramScoreSet([
            ("line_score", line_score), ("isotopic_fit", isotopic_fit),
            ("spacing_fit", spacing_fit), ("charge_count", charge_count)
        ])

    def logitscore(self, chromatogram):
        score = logitsum(self.compute_scores(chromatogram))
        return score


class ChromatogramScorer(ScorerBase):
    def __init__(self, shape_fitter_type=AdaptiveMultimodalChromatogramShapeFitter,
                 isotopic_fitter_type=IsotopicPatternConsistencyFitter,
                 charge_scoring_model=UniformChargeStateScoringModel(),
                 spacing_fitter_type=ChromatogramSpacingFitter,
                 adduct_scoring_model=None, *models):
        super(ChromatogramScorer, self).__init__()
        self.add_feature(shape_fitter_type)
        self.add_feature(isotopic_fitter_type)
        self.add_feature(spacing_fitter_type)
        self.add_feature(charge_scoring_model)
        self.add_feature(adduct_scoring_model)
        for model in models:
            self.add_feature(model)

    def compute_scores(self, chromatogram):
        scores = []
        for model in self.features():
            scores.append((model.get_feature_type(), model.score(chromatogram)))
        return ChromatogramScoreSet(scores)

    def clone(self):
        dup = self.__class__()
        for feature in self.features():
            dup.add_feature(feature)
        return dup


class ModelAveragingScorer(ScorerBase):
    def __init__(self, models, weights=None):
        if weights is None:
            weights = [1.0 for i in range(len(models))]
        self.models = models
        self.weights = weights

        self.prepare_weights()

    def prepare_weights(self):
        a = np.array(self.weights)
        a /= a.sum()
        self.weights = a

    def compute_scores(self, chromatogram):
        score_set = []
        for model, weight in zip(self.models, self.weights):
            score = model.compute_scores(chromatogram)
            score_set.append(np.array(score) * weight)
        return scores(*sum(score_set, np.zeros_like(score_set[0])))

    def clone(self):
        return self.__class__(list(self.models), list(self.weights))


class CompositionDispatchScorer(ScorerBase):
    def __init__(self, rule_model_map, default_model=None):
        if default_model is None:
            default_model = ChromatogramScorer()
        self.rule_model_map = rule_model_map
        self.default_model = default_model

    def clone(self):
        return self.__class__({k: v.clone() for k, v in self.rule_model_map.items()}, self.default_model.clone())

    def get_composition(self, obj):
        if obj.composition is not None:
            if obj.glycan_composition is not None:
                return symbolic_composition(obj)
        return None

    def find_model(self, composition):
        if composition is None:
            return self.default_model
        for rule in self.rule_model_map:
            if rule(composition):
                return self.rule_model_map[rule]

        return self.default_model

    def compute_scores(self, chromatogram):
        composition = self.get_composition(chromatogram)
        model = self.find_model(composition)
        return model.compute_scores(chromatogram)


class ChromatogramSolution(object):
    _temp_score = 0.0

    def __init__(self, chromatogram, score=None, scorer=ChromatogramScorer(), internal_score=None):
        if internal_score is None:
            internal_score = score
        self.chromatogram = chromatogram
        self.scorer = scorer
        self.internal_score = internal_score
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
        try:
            self.internal_score = self.score = self.scorer.score(self.chromatogram)
        except ZeroDivisionError:
            self.internal_score = self.score = 0.0

    def score_components(self):
        return self.scorer.compute_scores(self.chromatogram)

    @property
    def logitscore(self):
        return self.scorer.logitscore(self.chromatogram)

    def get_chromatogram(self):
        return self.chromatogram

    def clone(self):
        data = self.chromatogram.clone()
        return self.__class__(data, self.score, self.scorer, self.internal_score)

    def __repr__(self):
        return "ChromatogramSolution(%s, %0.4f, %d, %0.4f)" % (
            self.chromatogram.composition, self.chromatogram.neutral_mass,
            self.chromatogram.n_charge_states, self.score)

    def __getitem__(self, i):
        return self.chromatogram[i]

    def __dir__(self):
        return list(set(('compute_score', 'score_components')) | set(dir(self.chromatogram)))


ChromatogramInterface.register(ChromatogramSolution)
