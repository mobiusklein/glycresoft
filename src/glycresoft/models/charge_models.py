from glycresoft.scoring.base import (
    CompositionDispatchingModel,
    is_sialylated,
    DummyFeature)

from glycresoft.scoring import (
    MassScalingChargeStateScoringModel, ChromatogramScorer)

from .utils import make_model_loader
from .features import register_feature


load_model = make_model_loader(MassScalingChargeStateScoringModel)


SialylatedChargeModel = load_model("sialylated_charge_model")
UnsialylatedChargeModel = load_model("unsialylated_charge_model")

GeneralizedChargeScoringModel = CompositionDispatchingModel({
    is_sialylated: SialylatedChargeModel,
    lambda x: not is_sialylated(x): UnsialylatedChargeModel
}, SialylatedChargeModel)


GeneralScorer = ChromatogramScorer()
GeneralScorer.add_feature(GeneralizedChargeScoringModel)


register_feature("null_charge", DummyFeature("null_charge_model", "charge_count"))
