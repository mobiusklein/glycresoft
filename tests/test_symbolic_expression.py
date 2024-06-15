import unittest

from glypy.composition import composition_transform
from glypy.structure import glycan_composition

from glycresoft import symbolic_expression


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
        ctx = symbolic_expression.SymbolContext({"X": 5, "Z": 3})
        expected = 36
        self.assertEqual(ex.evaluate(ctx), expected)

    def test_nested_sub_expression(self):
        expr = symbolic_expression.parse_expression("(2(x + 5)) + 5")
        assert expr.evaluate({'x': 0}) == 15
        assert expr.evaluate({"x": 5}) == 25

    def test_nested_simplify(self):
        expr = symbolic_expression.parse_expression("(((x + 5)))")
        assert isinstance(expr.expr, symbolic_expression.ExpressionNode)
        expr = symbolic_expression.parse_expression("((((2(x + 5)))))")
        assert expr.coefficient == 2
        assert isinstance(expr.expr, symbolic_expression.ExpressionNode)


if __name__ == '__main__':
    unittest.main()
