from glycan_profiling.scoring import (
    MassScalingAdductScoringModel)

from .utils import make_model_loader
from .features import register_feature


load_model = make_model_loader(MassScalingAdductScoringModel)

AmmoniumAdductFeature = load_model("ammonium_adduct_model")


register_feature("permethylated_ammonium_adducts", AmmoniumAdductFeature)
