from StringIO import StringIO
from scoring import MassScalingChargeStateScoringModel, ChromatogramScorer, CompositionDispatchScorer
from glypy.composition import glycan_composition

_UnsialylatedNGlycanChargeScoringModel = MassScalingChargeStateScoringModel.load(StringIO('''
{
    "neighborhood_width": 50.0,
    "table": {
        "1500.0": {
            "-1": 0.003311258278145696,
            "-2": 0.9933774834437088,
            "-3": 0.003311258278145696
        },
        "2000.0": {
            "-1": 0.003311258278145696,
            "-2": 0.9933774834437088,
            "-3": 0.003311258278145696
        },
        "2500.0": {
            "-1": 0.003311258278145696,
            "-2": 0.9933774834437088,
            "-3": 0.003311258278145696
        },
        "3000.0": {
            "-1": 0.0019960079840319364,
            "-2": 0.5988023952095809,
            "-3": 0.3992015968063872
        },
        "3500.0": {
            "-1": 0.004950495049504951,
            "-2": 0.9900990099009903,
            "-3": 0.004950495049504951
        },
        "4000.0": {
            "-1": 0.00980392156862745,
            "-2": 0.00980392156862745,
            "-3": 0.9803921568627451
        }
    }
}
'''))


UnsialylatedNGlycanScorer = ChromatogramScorer(charge_scoring_model=_UnsialylatedNGlycanChargeScoringModel)


_SialylatedNGlycanChargeScoringModel = MassScalingChargeStateScoringModel.load(StringIO('''
{
    "table": {
        "3000.0": {
            "-2": 0.5828476269775188,
            "-4": 0.00041631973355537054,
            "-3": 0.41631973355537055,
            "-1": 0.00041631973355537054
        },
        "2250.0": {
            "-2": 0.6622516556291391,
            "-4": 0.0033112582781456954,
            "-3": 0.33112582781456956,
            "-1": 0.0033112582781456954
        },
        "4500.0": {
            "-1": 0.00041631973355537054,
            "-4": 0.41631973355537055,
            "-3": 0.5828476269775188,
            "-2": 0.00041631973355537054
        },
        "5250.0": {
            "-1": 0.0016611295681063125,
            "-4": 0.33222591362126247,
            "-3": 0.6644518272425249,
            "-2": 0.0016611295681063125
        },
        "3750.0": {
            "-2": 0.3998857469294487,
            "-4": 0.14281633818908884,
            "-3": 0.4570122822050843,
            "-1": 0.00028563267637817766
        }
    },
    "neighborhood_width": 75.0
}
'''))

SialylatedNGlycanScorer = ChromatogramScorer(charge_scoring_model=_SialylatedNGlycanChargeScoringModel)

neuac = glycan_composition.FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")


def is_sialylated(composition):
    return composition[neuac] > 0

rule_map = {
    is_sialylated: SialylatedNGlycanScorer,
    lambda x: not is_sialylated(x): UnsialylatedNGlycanScorer
}

GeneralScorer = CompositionDispatchScorer(rule_map, SialylatedNGlycanScorer)
