from collections import namedtuple
from operator import mul
try:
    reduce
except:
    from functools import reduce

import numpy as np

from .shape_fitter import ChromatogramShapeFitter
from .spacing_fitter import ChromatogramSpacingFitter
from .charge_state import UniformChargeStateScoringModel
from .isotopic_fit import IsotopicPatternConsistencyFitter

from glypy.composition.glycan_composition import FrozenGlycanComposition

from glycan_profiling.chromatogram_tree import ChromatogramInterface


epsilon = 1e-6

scores = namedtuple("scores", ["line_score", "isotopic_fit", "spacing_fit", "charge_count"])


def prod(*x):
    return reduce(mul, x, 1)


class ChromatogramScorer(object):
    def __init__(self, shape_fitter_type=ChromatogramShapeFitter,
                 isotopic_fitter_type=IsotopicPatternConsistencyFitter,
                 charge_scoring_model=UniformChargeStateScoringModel(),
                 spacing_fitter_type=ChromatogramSpacingFitter):
        self.shape_fitter_type = shape_fitter_type
        self.isotopic_fitter_type = isotopic_fitter_type
        self.spacing_fitter_type = spacing_fitter_type
        self.charge_scoring_model = charge_scoring_model

    def compute_scores(self, chromatogram):
        line_score = max(1 - self.shape_fitter_type(chromatogram).line_test, epsilon)
        isotopic_fit = max(1 - self.isotopic_fitter_type(chromatogram).mean_fit, epsilon)
        spacing_fit = max(1 - self.spacing_fitter_type(chromatogram).score * 2, epsilon)
        charge_count = self.charge_scoring_model.score(chromatogram)
        return scores(line_score, isotopic_fit, spacing_fit, charge_count)

    def score(self, chromatogram):
        score = reduce(mul, self.compute_scores(chromatogram), 1.0)
        return score


class ModelAveragingScorer(object):
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

    def score(self, chromatogram):
        return prod(*self.compute_scores(chromatogram))


class CompositionDispatchScorer(object):
    def __init__(self, rule_model_map, default_model=None):
        if default_model is None:
            default_model = ChromatogramScorer()
        self.rule_model_map = rule_model_map
        self.default_model = default_model

    def get_composition(self, obj):
        if isinstance(obj.composition, basestring):
            composition = FrozenGlycanComposition.parse(obj.composition)
        else:
            composition = obj.composition
        return composition

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

    def score(self, chromatogram):
        score = reduce(mul, self.compute_scores(chromatogram), 1.0)
        return score


class ChromatogramSolution(object):
    _temp_score = 0.0

    def __init__(self, chromatogram, score=None, scorer=ChromatogramScorer()):
        self.chromatogram = chromatogram
        self.scorer = scorer
        self.internal_score = self.score = score

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

    def __repr__(self):
        return "ChromatogramSolution(%s, %0.4f, %d, %0.4f)" % (
            self.chromatogram.composition, self.chromatogram.neutral_mass,
            self.chromatogram.n_charge_states, self.score)

    def __getitem__(self, i):
        return self.chromatogram[i]

    def __dir__(self):
        return set(('compure_score', 'score_components')) | set(dir(self.chromatogram))


ChromatogramInterface.register(ChromatogramSolution)
