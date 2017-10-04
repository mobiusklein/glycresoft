from glycan_profiling.scoring import (
    DummyFeature,
    MassScalingAdductScoringModel)

from glycan_profiling.scoring.base import (
    degree_of_sialylation,
    CompositionDispatchingModel)

from .utils import make_model_loader
from .features import register_feature


load_model = make_model_loader(MassScalingAdductScoringModel)

AmmoniumAdductFeature = load_model("ammonium_adduct_model")
MethylLossFeature = load_model("methyl_loss_adduct_model")


register_feature("permethylated_ammonium_adducts", AmmoniumAdductFeature)
register_feature("methyl_loss", MethylLossFeature)


class PermethylatedAmmoniumAndMethylLossModel(DummyFeature):
    def __init__(self, boosting=True):
        DummyFeature.__init__(
            self, name=self.__class__.__name__, feature_type="adduct_score")
        self.ammonium = AmmoniumAdductFeature
        self.methyl_loss = MethylLossFeature
        self.boosting = boosting

    def score(self, chromatogram, *args, **kwargs):
        methyl_score = self.methyl_loss.score(chromatogram)
        # the methyl loss scoring model is < 0.2 if only a loss
        # was matched, 0.5 if only unmodified was matched, and
        # > 0.5 if both unmodified and a loss was matched.
        #
        # to make this scorer pass through directly to the ammonium
        # model when there was no methyl loss detected, either the
        # case where unmodified is detecteded should map to essentially
        # 1.0 or the case where no loss is registered
        methyl_score *= 2
        if not self.boosting:
            if methyl_score > 1:
                methyl_score = 1
        # there was no methyl loss
        if methyl_score < 1e-3:
            methyl_score = 1
        # elif methyl_score > 1:
        #     methyl_score = 1
        score = methyl_score * self.ammonium.score(chromatogram)
        if score > 1:
            score = 1 - 1e-3
        return score


PermethylatedAmmoniumAndMethylLossFeature = PermethylatedAmmoniumAndMethylLossModel()

register_feature(
    "permethylated_ammonium_adducts_methyl_loss",
    PermethylatedAmmoniumAndMethylLossFeature)


AsialoFormateAdductFeature = load_model("asialo_formate_adduct_model")
MonosialoFormateAdductFeature = load_model("monosialo_formate_adduct_model")
DisialoFormateAdductFeature = load_model("disialo_formate_adduct_model")
TrisialoFormateAdductFeature = load_model("trisialo_formate_adduct_model")
TetrasialoFormateAdductFeature = load_model("tetrasialo_formate_adduct_model")

_GeneralizedFormateAdductModelTable = CompositionDispatchingModel({
    lambda x: degree_of_sialylation(x) == 0: AsialoFormateAdductFeature,
    lambda x: degree_of_sialylation(x) == 1: MonosialoFormateAdductFeature,
    lambda x: degree_of_sialylation(x) == 2: DisialoFormateAdductFeature,
    lambda x: degree_of_sialylation(x) == 3: TrisialoFormateAdductFeature,
    lambda x: degree_of_sialylation(x) > 3: TetrasialoFormateAdductFeature
}, AsialoFormateAdductFeature)


class GeneralizedFormateAdductModel(DummyFeature):
    def __init__(self):
        DummyFeature.__init__(
            self, name=self.__class__.__name__, feature_type="adduct_score")
        self.table = _GeneralizedFormateAdductModelTable

    def score(self, chromatogram, *args, **kwargs):
        score = self.table.score(chromatogram, *args, **kwargs)
        # This feature can easily be 1.0 under certain conditions, which
        # leads to undesirable score inflation. The maximum contribution
        # this should have is capped to 0.7 (logit(0.7) = 0.8472) which
        # should not lead to many more assignments being retained while
        # still penalizing inappropriate formate adduction states
        return min(score, 0.7)


GeneralizedFormateAdductFeature = GeneralizedFormateAdductModel()


register_feature("formate_adduct_model", GeneralizedFormateAdductFeature)
