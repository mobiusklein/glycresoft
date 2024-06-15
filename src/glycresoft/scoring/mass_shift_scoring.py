import json
import warnings

from collections import defaultdict
from io import StringIO

from glypy.composition import Composition, formula

from glycresoft.chromatogram_tree import Unmodified, MassShift, CompoundMassShift
from .base import (
    MassScalingCountScoringModel,
    UniformCountScoringModelBase,
    ScoringFeatureBase)


class MassShiftScoringModelBase(ScoringFeatureBase):
    feature_type = "mass_shift_score"

    def __init__(self, mass_shift_types=None):
        if mass_shift_types is not None:
            self.mass_shift_types = set(mass_shift_types) | {Unmodified}
        else:
            self.mass_shift_types = None

    def get_state_count(self, chromatogram):
        if self.mass_shift_types is None:
            return len(chromatogram.mass_shifts)
        else:
            return len(
                [t for t in self.mass_shift_types
                 if t in chromatogram.mass_shifts])

    def get_states(self, chromatogram):
        if self.mass_shift_types is None:
            return chromatogram.mass_shifts
        else:
            return [t for t in self.mass_shift_types
                    # if t in chromatogram.mass_shifts
                    if [k for k in chromatogram.mass_shifts if k == t or k.composed_with(t)]
                    ]

    def get_signal_proportions(self, chromatogram):
        fractions = chromatogram.mass_shift_signal_fractions()
        total = sum(fractions.values())
        if self.mass_shift_types is None:
            return {k: v / total for k, v in fractions.items()}
        else:
            proportions = defaultdict(float)
            for t in self.mass_shift_types:
                for k, f in fractions.items():
                    if k.composed_with(t):
                        proportions[t] += f
            total = sum(proportions.values())
            return {k: v / total for k, v in proportions.items()}


class UniformMassShiftScoringModel(MassShiftScoringModelBase, UniformCountScoringModelBase):
    pass


uniform_model = UniformMassShiftScoringModel()


class MassScalingMassShiftScoringModel(MassShiftScoringModelBase, MassScalingCountScoringModel):
    def __init__(self, table, mass_shift_types=None, neighborhood_width=100., fit_information=None):
        MassShiftScoringModelBase.__init__(self, mass_shift_types)
        MassScalingCountScoringModel.__init__(self, table, neighborhood_width)
        self.fit_information = fit_information or {}

    def transform_state(self, state):
        return state.name

    def handle_missing_neighborhood(self, chromatogram, neighborhood, *args, **kwargs):
        warnings.warn(
            ("%f was not found for this mass_shift "
             "scoring model. Defaulting to uniform model") % neighborhood)
        return uniform_model.score(chromatogram, *args, **kwargs)

    def handle_missing_bin(self, chromatogram, bins, key, neighborhood, *args, **kwargs):
        warnings.warn("%s not found for this mass range (%f). Using bin average (%r)" % (
            key, neighborhood, chromatogram.mass_shifts))
        return 0

    def clone(self):
        text_buffer = StringIO()
        self.dump(text_buffer)
        text_buffer.seek(0)
        return self.load(text_buffer)

    def _serialize_mass_shift(self, mass_shift):
        return {"name": mass_shift.name, "composition": formula(mass_shift.composition)}

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

    def _serialize_mass_shift_types(self):
        if self.mass_shift_types is None:
            return None
        mass_shift_types = []
        for mass_shift in self.mass_shift_types:
            if isinstance(mass_shift, CompoundMassShift):
                mass_shift_types.append(self._serialize_compound_mass_shift(mass_shift))
            elif isinstance(mass_shift, MassShift):
                mass_shift_types.append(self._serialize_mass_shift(mass_shift))
        return mass_shift_types

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
    def _deserialize_mass_shift_types(cls, data):
        if data is None:
            return None
        mass_shift_types = []
        memo = {
            Unmodified.name: Unmodified
        }
        for entry in data:
            if "counts" in entry:
                mass_shift_types.append(cls._deserialize_compound_mass_shift(entry, memo))
            else:
                mass_shift_types.append(cls._deserialize_mass_shift(entry, memo))
        return mass_shift_types

    def dump(self, file_obj):
        json.dump(
            {
                "neighborhood_width": self.neighborhood_width,
                "table": self.table,
                "mass_shift_types": self._serialize_mass_shift_types(),
                "fit_information": self.fit_information
            },
            file_obj, indent=4, sort_keys=True)

    @classmethod
    def load(cls, file_obj):
        data = json.load(file_obj)
        table = data.pop("table")
        width = float(data.pop("neighborhood_width"))
        mass_shift_types = data.pop("mass_shift_types")
        if mass_shift_types is not None:
            mass_shift_types = cls._deserialize_mass_shift_types(mass_shift_types)
        if isinstance(mass_shift_types, list):
            mass_shift_types = set(mass_shift_types)

        def numeric_keys(table, dtype=float, convert_value=lambda x: x):
            return {abs(dtype(k)): convert_value(v) for k, v in table.items()}

        table = numeric_keys(table)

        return cls(table=table, neighborhood_width=width, mass_shift_types=mass_shift_types)

    @classmethod
    def fit(cls, observations, missing=0.01, mass_shift_types=None, neighborhood_width=100.,
            scale_unmodified=0.25):
        if not scale_unmodified:
            scale_unmodified = 1
        bins = defaultdict(lambda: defaultdict(float))

        self = cls({}, mass_shift_types=mass_shift_types,
                   neighborhood_width=neighborhood_width)

        fit_info = {
            "scale_unmodified": scale_unmodified,
            "missing": missing,
            "track": defaultdict(lambda: defaultdict(list)),
            "count": defaultdict(int)
        }

        all_keys = set()
        for sol in observations:
            neighborhood = self.neighborhood_of(sol.neutral_mass)
            fit_info['count'][neighborhood] += 1
            for c in self.get_states(sol):
                key = self.transform_state(c)
                all_keys.add(key)
                val = sol.total_signal_for(c) / (sol.total_signal)
                bins[neighborhood][key] += val
                fit_info['track'][neighborhood][key].append(val)

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

        return cls(model_table, mass_shift_types=mass_shift_types, neighborhood_width=neighborhood_width,
                   fit_information=fit_info)


MassScalingAdductScoringModel = MassScalingMassShiftScoringModel
