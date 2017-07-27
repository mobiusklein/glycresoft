import json
import warnings

from collections import defaultdict
from io import StringIO

from glypy.composition import Composition, formula

from glycan_profiling.chromatogram_tree import Unmodified, MassShift, CompoundMassShift
from .base import (
    MassScalingCountScoringModel,
    UniformCountScoringModelBase,
    ScoringFeatureBase)


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

    def get_signal_proportions(self, chromatogram):
        fractions = chromatogram.adduct_signal_fractions()
        total = sum(fractions.values())
        if self.adduct_types is None:
            return {k: v / total for k, v in fractions.items()}
        else:
            proportions = defaultdict(float)
            for t in self.adduct_types:
                for k, f in fractions.items():
                    if k.composed_with(t):
                        proportions[t] += f
            total = sum(proportions.values())
            return {k: v / total for k, v in proportions.items()}


class UniformAdductScoringModel(AdductScoringModelBase, UniformCountScoringModelBase):
    pass


uniform_model = UniformAdductScoringModel()


class MassScalingAdductScoringModel(AdductScoringModelBase, MassScalingCountScoringModel):
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

    def clone(self):
        text_buffer = StringIO()
        self.dump(text_buffer)
        text_buffer.seek(0)
        return self.load(text_buffer)

    def _serialize_mass_shift(self, mass_shift):
        return {"name": mass_shift.name, "composition": dict(mass_shift.composition)}

    def _serialize_compound_mass_shift(self, mass_shift):
        return {
            "name": mass_shift.name,
            "composition": formula(mass_shift.composition),
            "counts": {
                k.name: v for k, v in mass_shift.counts.items()
            },
            "definitions": {
                k.name: formula(k.composition) for k, v in mass_shift.counts.items()
            }
        }

    def _serialize_adduct_types(self):
        if self.adduct_types is None:
            return None
        adduct_types = []
        for adduct in self.adduct_types:
            if isinstance(adduct, CompoundMassShift):
                adduct_types.append(self._serialize_compound_mass_shift(adduct))
            elif isinstance(adduct, MassShift):
                adduct_types.append(self._serialize_mass_shift(adduct))
        return adduct_types

    @classmethod
    def _deserialize_mass_shift(cls, data, memo=None):
        if memo is not None:
            if data['name'] in memo:
                return memo[data['name']]
        inst = MassShift(data['name'], Composition(str(data['composition'])))
        if memo is not None:
            memo[inst.name] = inst
        return inst

    @classmethod
    def _deserialize_compound_mass_shift(cls, data, memo=None):
        if memo is not None:
            if data['name'] in memo:
                return memo[data['name']]
        definitions = {}
        for k, v in data['definitions'].items():
            definitions[k] = cls._deserialize_mass_shift({"name": k, "composition": v})

        components = {}
        for k, v in data['counts'].items():
            components[definitions[k]] = v
        inst = CompoundMassShift(components)

        if memo is not None:
            memo[inst.name] = inst
        return inst

    @classmethod
    def _deserialize_adduct_types(cls, data):
        if data is None:
            return None
        adduct_types = []
        memo = {
            Unmodified.name: Unmodified
        }
        for entry in data:
            if "counts" in entry:
                adduct_types.append(cls._deserialize_compound_mass_shift(entry, memo))
            else:
                adduct_types.append(cls._deserialize_mass_shift(entry, memo))
        return adduct_types

    def dump(self, file_obj):
        json.dump(
            {
                "neighborhood_width": self.neighborhood_width,
                "table": self.table,
                "adduct_types": self._serialize_adduct_types(),
            },
            file_obj, indent=4, sort_keys=True)

    @classmethod
    def load(cls, file_obj):
        data = json.load(file_obj)
        table = data.pop("table")
        width = float(data.pop("neighborhood_width"))
        adduct_types = data.pop("adduct_types")
        if adduct_types is not None:
            adduct_types = cls._deserialize_adduct_types(adduct_types)
        if isinstance(adduct_types, list):
            adduct_types = set(adduct_types)

        def numeric_keys(table, dtype=float, convert_value=lambda x: x):
            return {abs(dtype(k)): convert_value(v) for k, v in table.items()}

        table = numeric_keys(table)

        return cls(table=table, neighborhood_width=width, adduct_types=adduct_types)

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
