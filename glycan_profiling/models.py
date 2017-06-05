from io import StringIO
from glypy.composition import glycan_composition

from glycan_profiling.scoring.adduct_scoring import AdductMassScalingCountScoringModel
from glycan_profiling.scoring import MassScalingChargeStateScoringModel, ChromatogramScorer, CompositionDispatchScorer


_UnsialylatedNGlycanChargeScoringModel = MassScalingChargeStateScoringModel.load(StringIO(u'''
{
    "neighborhood_width": 50.0,
    "table": {
        "500.0": {
            "-1": 0.95,
            "-2": 0.047,
            "-3": 0.003
        },
        "1000.0": {
            "-1": 0.75,
            "-2": 0.247,
            "-3": 0.003
        },
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


UnsialylatedNGlycanScorer = ChromatogramScorer(
    charge_scoring_model=_UnsialylatedNGlycanChargeScoringModel)


_SialylatedNGlycanChargeScoringModel = MassScalingChargeStateScoringModel.load(StringIO(u'''
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


SialylatedNGlycanScorer = ChromatogramScorer(
    charge_scoring_model=_SialylatedNGlycanChargeScoringModel)


AmmoniumAdductFeature = AdductMassScalingCountScoringModel.load(StringIO(u'''
{
    "adduct_types": null,
    "neighborhood_width": 50.0,
    "table": {
        "2000.0": {
            "Ammonium": 0.7349836056183142,
            "Ammonium * 2": 0.015211047404581431,
            "Ammonium * 3": 0.003048305724334401,
            "Unmodified": 0.23456381835543236
        },
        "2500.0": {
            "Ammonium": 0.40227488800899536,
            "Ammonium * 2": 0.048604490183706085,
            "Ammonium * 3": 0.00546273755451629,
            "Unmodified": 0.5380851121074653
        },
        "3000.0": {
            "Ammonium": 0.1968631651567533,
            "Ammonium * 2": 0.19501196930021236,
            "Ammonium * 3": 0.03852735206742863,
            "Unmodified": 0.5618272287554293
        },
        "3500.0": {
            "Ammonium": 0.3837365572105348,
            "Ammonium * 2": 0.005180331547402133,
            "Ammonium * 3": 0.005180331547402133,
            "Unmodified": 0.5851814535050524
        },
        "4000.0": {
            "Ammonium": 0.2943977260022403,
            "Ammonium * 2": 0.3551123672201971,
            "Ammonium * 3": 0.0016672714898488428,
            "Unmodified": 0.3421535493283184
        },
        "4500.0": {
            "Ammonium": 0.2579939707512655,
            "Ammonium * 2": 0.007571222950942212,
            "Ammonium * 3": 0.004230328795848274,
            "Unmodified": 0.7132831623185508
        },
        "5000.0": {
            "Ammonium": 0.2487743736514156,
            "Ammonium * 2": 0.000586498800846463,
            "Ammonium * 3": 0.004269851530604238,
            "Unmodified": 0.7292898698947167
        }
    }
}
'''))


neuac = glycan_composition.FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")
neugc = glycan_composition.FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")
neu = glycan_composition.FrozenMonosaccharideResidue.from_iupac_lite("Neu")


def is_sialylated(composition):
    return (composition[neuac] + composition[neugc] + composition[neu]) > 0


rule_map = {
    is_sialylated: SialylatedNGlycanScorer,
    lambda x: not is_sialylated(x): UnsialylatedNGlycanScorer
}


GeneralScorer = CompositionDispatchScorer(rule_map, SialylatedNGlycanScorer)
