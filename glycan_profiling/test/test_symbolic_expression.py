import unittest

from glypy.composition import composition_transform
from glypy.structure import glycan_composition

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

    def test_complex_expression(self):
        ex = symbolic_expression.parse_expression("X + 1 + abs(-2Z) * 5")
        ctx = ctx = symbolic_expression.SymbolContext({"X": 5, "Z": 3})
        expected = 36
        self.assertEqual(ex.evaluate(ctx), expected)


if __name__ == '__main__':
    unittest.main()
