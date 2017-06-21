import unittest

from glypy.composition import glycan_composition, composition_transform

from glycan_profiling import symbolic_expression


class SymbolicExpressionTest(unittest.TestCase):
    def test_normalize_glycan_composition(self):
        base = glycan_composition.GlycanComposition.parse("{Hex:6; HexNAc:5; Neu5Ac:3}")
        deriv = composition_transform.derivatize(
            glycan_composition.GlycanComposition.parse(
                "{Hex:6; HexNAc:5; Neu5Ac:3}"), "methyl")

        normd_symbol = symbolic_expression.GlycanSymbolContext(deriv)
        normd_composition = glycan_composition.GlycanComposition.parse(normd_symbol.serialize())

        self.assertEqual(base, normd_composition)


if __name__ == '__main__':
    unittest.main()
