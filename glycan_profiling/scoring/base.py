import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from glycan_profiling import symbolic_expression
from glycopeptidepy import HashableGlycanComposition
from glypy.composition.glycan_composition import FrozenMonosaccharideResidue as FMR


epsilon = 1e-6


def symbolic_composition(obj):
    """Extract the glycan composition from *obj* and converts it
    into an instance of :class:`.GlycanSymbolContext`

    The extraction process first attempts to access *obj.glycan_composition*
    but on an :class:`AttributeError` calls :meth:`.HashableGlycanComposition.parse`
    on *obj*. The resulting :class:`.GlycanComposition` object is then passed
    to :class:`.GlycanSymbolContext`

    Parameters
    ----------
    obj : object
        The object to convert

    Returns
    -------
    GlycanSymbolContext
    """
    try:
        composition = obj.glycan_composition
    except AttributeError:
        composition = HashableGlycanComposition.parse(obj)
    composition = symbolic_expression.GlycanSymbolContext(composition)
    return composition


class ScoringFeatureBase(object):
    """A base class for a type that either on instantiation
    or on invokation of :meth:`score` will produce a confidence
    metric for a (possibly annotated) chromatogram
    """
    name = None
    feature_type = None

    @classmethod
    def get_feature_type(self):
        return self.feature_type

    @classmethod
    def get_feature_name(self):
        name = getattr(self, "name", None)
        if name is None:
            return "%s:%s" % (self.get_feature_type(), self.__name__)
        else:
            return name

    @classmethod
    def reject(self, score_components):
        score = score_components[self.get_feature_type()]
        return score < 0.15

    @classmethod
    def configure(cls, analysis_data):
        return {}


class DummyFeature(ScoringFeatureBase):
    def __init__(self, name, feature_type):
        self.name = name
        self.feature_type = feature_type

    def score(self, chromatogram, *args, **kwargs):
        return 0.985

    def get_feature_type(self):
        return self.feature_type

    def get_feature_name(self):
        name = getattr(self, "name", None)
        if name is None:
            name = self.__name__
        return "%s:%s" % (self.get_feature_type(), name)

    def reject(self, score_components):
        score = score_components[self.feature_type]
        return score < 0.15

    def clone(self):
        return self


@add_metaclass(ABCMeta)
class DiscreteCountScoringModelBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, chromatogram, *args, **kwargs):
        return 0

    def dump(self, file_obj):
        pickle.dump(self, file_obj)

    @classmethod
    def load(cls, file_obj):
        return pickle.load(file_obj)

    @abstractmethod
    def get_state_count(self, chromatogram):
        # Returns the number of states
        raise NotImplementedError()

    @abstractmethod
    def get_states(self, chromatogram):
        # Returns a list enumerating the states
        # this entity was observed in
        raise NotImplementedError()

    @abstractmethod
    def get_signal_proportions(self, chromatogram):
        # Returns a mapping from state to proportion of
        # total signal observed in state
        raise NotImplementedError()

    def reject(self, score_components):
        score = score_components[self.feature_type]
        return score < 0.15


class UniformCountScoringModelBase(DiscreteCountScoringModelBase):
    def score(self, chromatogram, *args, **kwargs):
        return min(0.4 * self.get_state_count(chromatogram), 1.0) - epsilon


def decay(x, step=0.4, rate=1.5):
    v = 0
    for i in range(x):
        v += (step / (i + rate))
    return v


class DecayRateCountScoringModelBase(DiscreteCountScoringModelBase):
    def __init__(self, step=0.4, rate=1.5):
        self.step = step
        self.rate = rate

    def score(self, chromatogram, *args, **kwargs):
        k = self.get_state_count(chromatogram)
        return decay(k, self.step, self.rate) - epsilon

    def clone(self):
        return self.__class__(self.step, self.rate)


class LogarithmicCountScoringModelBase(DiscreteCountScoringModelBase):
    def __init__(self, steps=5):
        self.base = steps

    def _logarithmic_change_of_base(self, k):
        if k >= self.base:
            return 1.0
        return np.log(k) / np.log(self.base)

    def score(self, chromatogram, *args, **kwargs):
        # Ensure k > 1 so that the value is greater than 0.0
        # as `log(1) = 0`
        k = self.get_state_count(chromatogram) + 1
        return self._logarithmic_change_of_base(k) - epsilon

    def clone(self):
        return self.__class__(self.base)


def ones(x):
    return (x - (np.floor(x / 10.) * 10))


def neighborhood_of(x, scale=100.):
    n = x / scale
    up = ones(n) > 5
    if up:
        neighborhood = (np.floor(n / 10.) + 1) * 10
    else:
        neighborhood = (np.floor(n / 10.) + 1) * 10
    return neighborhood * scale


class MassScalingCountScoringModel(DiscreteCountScoringModelBase):
    def __init__(self, table, neighborhood_width=100.):
        self.table = table
        self.neighborhood_width = neighborhood_width

    def neighborhood_of(self, mass):
        n = mass / self.neighborhood_width
        up = ones(n) > 5
        if up:
            neighborhood = (np.floor(n / 10.) + 1) * 10
        else:
            neighborhood = (np.floor(n / 10.) + 1) * 10
        return neighborhood * self.neighborhood_width

    def get_neighborhood_key(self, neutral_mass):
        neighborhood = self.neighborhood_of(neutral_mass)
        return neighborhood

    def handle_missing_neighborhood(self, chromatogram, neighborhood, *args, **kwargs):
        return 0

    def handle_missing_bin(self, chromatogram, bins, key, neighborhood, *args, **kwargs):
        return sum(bins.values()) / float(len(bins))

    def transform_state(self, state):
        return state

    def score(self, chromatogram, *args, **kwargs):
        total = 0.
        neighborhood = self.get_neighborhood_key(chromatogram.neutral_mass)
        if neighborhood not in self.table:
            return self.handle_missing_neighborhood(
                chromatogram, neighborhood, *args, **kwargs)

        bins = self.table[neighborhood]

        for state in self.get_states(chromatogram):
            state = self.transform_state(state)
            try:
                total += bins[state]
            except KeyError:
                total += self.handle_missing_bin(
                    chromatogram, bins, state, neighborhood, *args, **kwargs)
        total = max(min(total, 1.0) - epsilon, epsilon)
        return total

    def clone(self):
        return self.__class__(self.table, self.neighborhood_width)


class ProportionBasedMassScalingCountScoringModel(MassScalingCountScoringModel):

    def kullback_leibler_divergence(self, observed, expected):
        total = 0
        for key in expected:
            try:
                observed_frequency = observed[key]
            except KeyError:
                observed_frequency = 1e-3
            total += observed_frequency * (np.log(observed_frequency) - np.log(expected[key]))
        return total

    def expnorm_kl(self, observed, expected):
        return 1 - np.exp(-self.kullback_leibler_divergence(observed, expected))

    def score(self, chromatogram, *args, **kwargs):
        neighborhood = self.get_neighborhood_key(chromatogram.neutral_mass)
        if neighborhood not in self.table:
            return self.handle_missing_neighborhood(
                chromatogram, neighborhood, *args, **kwargs)

        model_bin = self.table[neighborhood]
        observed_dist = dict()

        for key, proportion in self.get_signal_proportions(chromatogram):
            key = self.transform_state(key)
            observed_dist[key] = proportion
        score = self.expnorm_kl(observed_dist, model_bin)
        score = max(min(score, 1.0) - epsilon, epsilon)
        return score


class CompositionDispatchingModel(ScoringFeatureBase):
    def __init__(self, rule_model_map, default_model):
        self.rule_model_map = rule_model_map
        self.default_model = default_model

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

    def score(self, chromatogram, *args, **kwargs):
        composition = self.get_composition(chromatogram)
        model = self.find_model(composition)
        return model.score(chromatogram, *args, **kwargs)

    def get_feature_type(self):
        return self.default_model.get_feature_type()

    def get_feature_name(self):
        return "%s:%s(%s)" % (
            self.get_feature_type(), self.__class__.__name__,
            ", ".join([v.get_feature_name()
                       for v in self.rule_model_map.values()]))

    def reject(self, score_components):
        score = score_components[self.get_feature_type()]
        return score < 0.15

    def clone(self):
        return self.__class__(self.rule_model_map, self.default_model)


neuac = FMR.from_iupac_lite("NeuAc")
neugc = FMR.from_iupac_lite("NeuGc")
neu = FMR.from_iupac_lite("Neu")


def is_sialylated(composition):
    return (composition[neuac] + composition[neugc] + composition[neu]) > 0


def degree_of_sialylation(composition):
    return (composition[neuac] + composition[neugc] + composition[neu])
