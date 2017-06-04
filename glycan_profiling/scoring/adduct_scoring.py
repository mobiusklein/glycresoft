import json
import warnings
from collections import defaultdict

from glycan_profiling.chromatogram_tree import Unmodified
from .base import (
    DiscreteCountScoringModelBase, MassScalingCountScoringModel,
    UniformCountScoringModelBase, ScoringFeatureBase)


class AdductScoringModelBase(ScoringFeatureBase):
    feature_type = "adduct_score"

    def __init__(self, adduct_types=None):
        if adduct_types is not None:
            self.adduct_types = set(adduct_types) | {Unmodified}
        else:
            self.adduct_types = None

    def get_state_count(self, chromatogram):
        if self.adduct_types is None:
            return len(chromatogram.adducts)
        else:
            return len(
                [t for t in self.adduct_types
                 if t in chromatogram.adducts])

    def get_states(self, chromatogram):
        if self.adduct_types is None:
            return chromatogram.adducts
        else:
            return [t for t in self.adduct_types
                    if t in chromatogram.adducts]


class UniformAdductScoringModel(AdductScoringModelBase, UniformCountScoringModelBase):
    pass


uniform_model = UniformAdductScoringModel()


class AdductMassScalingCountScoringModel(AdductScoringModelBase, MassScalingCountScoringModel):
    def __init__(self, table, adduct_types=None, neighborhood_width=100.):
        AdductScoringModelBase.__init__(self, adduct_types)
        MassScalingCountScoringModel.__init__(self, table, neighborhood_width)

    def transform_state(self, state):
        return state.name

    def handle_missing_neighborhood(self, chromatogram, neighborhood, *args, **kwargs):
        warnings.warn(
            ("%f was not found for this charge state "
             "scoring model. Defaulting to uniform model") % neighborhood)
        return uniform_model.score(chromatogram, *args, **kwargs)

    def handle_missing_bin(self, chromatogram, bins, key, neighborhood, *args, **kwargs):
        warnings.warn("%s not found for this mass range (%f). Using bin average (%r)" % (
            key, neighborhood, chromatogram.adducts))
        return sum(bins.values()) / float(len(bins))

    def save(self, file_obj):
        json.dump(
            {
                "neighborhood_width": self.neighborhood_width,
                "table": self.table,
                "adduct_types": self.adduct_types if self.adduct_types is None else list(
                    self.adduct_types)
            },
            file_obj, indent=4, sort_keys=True)

    @classmethod
    def load(cls, file_obj):
        data = json.load(file_obj)
        table = data.pop("table")
        width = float(data.pop("neighborhood_width"))
        adduct_types = data.pop("adduct_types")
        if isinstance(adduct_types, list):
            adduct_types = set(adduct_types)

        def numeric_keys(table, dtype=float, convert_value=lambda x: x):
            return {abs(dtype(k)): convert_value(v) for k, v in table.items()}

        table = numeric_keys(table)

        return cls(table=table, neighborhood_width=width)

    @classmethod
    def fit(cls, observations, missing=0.01, adduct_types=None, neighborhood_width=100.,
            scale_unmodified=0.25):
        if not scale_unmodified:
            scale_unmodified = 1
        bins = defaultdict(lambda: defaultdict(float))

        self = cls({}, adduct_types=adduct_types,
                   neighborhood_width=neighborhood_width)

        all_keys = set()
        for sol in observations:
            neighborhood = self.neighborhood_of(sol.neutral_mass)
            for c in self.get_states(sol):
                key = self.transform_state(c)
                all_keys.add(key)
                bins[neighborhood][key] += sol.total_signal_for(c) / (sol.total_signal)

        model_table = {}

        all_states = set()
        for level in bins.values():
            all_states.update(level.keys())

        for neighborhood, counts in bins.items():
            for c in all_states:
                if str(c) == Unmodified.name:
                    counts[c] *= scale_unmodified
                counts[str(c)] += missing
            total = sum(counts.values())
            entry = {k: v / total for k, v in counts.items()}
            model_table[neighborhood] = entry

        return cls(model_table, adduct_types=adduct_types, neighborhood_width=neighborhood_width)
