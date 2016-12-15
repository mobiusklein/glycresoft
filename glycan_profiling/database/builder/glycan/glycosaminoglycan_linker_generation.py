from glypy import monosaccharides, GlycanComposition
from glypy.structure import (
    Modification, Glycan, Substituent)


def gag_linkers():
    xyl = monosaccharides.Xyl
    gal = monosaccharides.Gal
    xyl.add_monosaccharide(gal, 3)
    gal2 = monosaccharides.Gal
    gal.add_monosaccharide(gal2, 3)
    hexa = monosaccharides.Hex
    hexa.add_modification(Modification.Acidic, 6)
    gal2.add_monosaccharide(hexa, 3)
    hexnac = monosaccharides.HexNAc
    hexa.add_monosaccharide(hexnac, 3)
    enhexa = monosaccharides.Hex
    enhexa.add_modification(Modification.Acidic, 6)
    enhexa.add_modification(Modification.en, 2)
    hexnac.add_monosaccharide(enhexa, 3)
    base_linker = Glycan(xyl).reindex()

    variants = [{"substituents": {
        5: "sulfate"
    }}, {
        "monosaccharides": {
            1: "Fuc"
        }
    }, {
        "monosaccharides": {
            1: "Fuc"
        },
        "substituents": {
            5: "sulfate"
        }
    }, {
        "monosaccharides": {
            1: "Fuc"
        },
        "substituents": {
            5: "sulfate",
            3: "sulfate"
        }
    }, {
        "substituents": {
            1: "phosphate"
        }
    }, {
        "substituents": {
            1: "phosphate",
            5: "sulfate"
        }
    }, {
        "substituents": {
            1: "phosphate",
            3: "sulfate",
            5: "sulfate"
        }
    }, {
        "substituents": {
            5: "sulfate"
        }
    }, {
        "substituents": {
            3: "sulfate",
            5: "sulfate"
        }
    }, {
        "monosaccharides": {
            2: "NeuAc"
        }
    }, {
        "monosaccharides": {
            2: "NeuAc"
        },
        "substituents": {
            5: "sulfate"
        }
    }, {
        "monosaccharides": {
            2: "NeuAc"
        },
        "substituents": {
            5: "sulfate",
            3: "sulfate"
        }
    }]

    linker_variants = [base_linker]

    for variant in variants:
        linker = base_linker.clone()

        substituent_variant = variant.get("substituents", {})
        for position, subst in substituent_variant.items():
            parent = linker.get(position)
            parent.add_substituent(Substituent(subst))

        monosacch_variant = variant.get("monosaccharides", {})
        for position, mono in monosacch_variant.items():
            parent = linker.get(position)
            child = monosaccharides[mono]
            parent.add_monosaccharide(child, -1)

        linker.reindex()
        linker_variants.append(linker)

    return linker_variants


def gag_linker_compositions():
    return [GlycanComposition.from_glycan(linker) for linker in gag_linkers()]
