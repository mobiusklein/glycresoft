import numpy as np
from collections import defaultdict
import warnings


class ChargeStateDistributionScoringModelBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, chromatogram, *args, **kwargs):
        return 0

    def save(self, file_obj):
        pass

    @classmethod
    def load(cls, file_obj):
        return cls()


class UniformChargeStateScoringModel(ChargeStateDistributionScoringModelBase):
    def score(self, chromatogram, *args, **kwargs):
        return min(0.4 * chromatogram.n_charge_states, 1.0)


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


def get_sign(num):
    if num > 0:
        return 1
    else:
        return -1


class MassScalingChargeStateScoringModel(ChargeStateDistributionScoringModelBase):
    def __init__(self, table, neighborhood_width=100.):
        self.table = table
        self.neighborhood_width = neighborhood_width

    def score(self, chromatogram, *args, **kwargs):
        total = 0.
        neighborhood = neighborhood_of(chromatogram.neutral_mass, self.neighborhood_width)
        if neighborhood not in self.table:
            warnings.warn(
                ("%f was not found for this charge state "
                 "scoring model. Defaulting to uniform model") % neighborhood)
            return UniformChargeStateScoringModel().score(chromatogram, *args, **kwargs)
        bins = self.table[neighborhood]

        bin_sign = get_sign(tuple(bins)[0])
        chrom_sign = get_sign(tuple(chromatogram.charge_states)[0])
        if bin_sign != chrom_sign:
            chrom_sign = -1
        else:
            chrom_sign = 1

        for charge in chromatogram.charge_states:
            charge = chrom_sign * charge
            try:
                total += bins[charge]
            except KeyError:
                warnings.warn("%d not found for this mass range (%f). Using bin average (%r, %r)" % (
                    charge, neighborhood, bin_sign, chromatogram.charge_states))
                total += sum(bins.values()) / float(len(bins))
        total = min(total, 1.0)
        return total

    @classmethod
    def fit(cls, observations, missing=0.01, neighborhood_width=100., ignore_singly_charged=False):
        bins = defaultdict(lambda: defaultdict(float))

        for sol in observations:
            neighborhood = neighborhood_of(sol.neutral_mass, neighborhood_width)
            for c in sol.charge_states:
                if ignore_singly_charged and abs(c) == 1:
                    continue
                bins[neighborhood][c] += 1

        model_table = {}

        all_states = set()
        for level in bins.values():
            all_states.update(level.keys())

        all_states.add(1 * (min(all_states) / abs(min(all_states))))

        for neighborhood, counts in bins.items():
            for c in all_states:
                if counts[c] == 0:
                    counts[c] = missing
            total = sum(counts.values())
            entry = {k: v / total for k, v in counts.items()}
            model_table[neighborhood] = entry

        return cls(model_table, neighborhood_width)

    def save(self, file_obj):
        import json
        json.dump(
            {"neighborhood_width": self.neighborhood_width, "table": self.table},
            file_obj, indent=4, sort_keys=True)

    @classmethod
    def load(cls, file_obj):
        import json
        data = json.load(file_obj)
        table = data.pop("table")
        width = float(data.pop("neighborhood_width"))

        def numeric_keys(table, dtype=float, convert_value=lambda x: x):
            return {dtype(k): convert_value(v) for k, v in table.items()}

        table = numeric_keys(table, convert_value=lambda x: numeric_keys(x, int))

        return cls(table=table, neighborhood_width=width)
