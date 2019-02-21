import unittest

from glycan_profiling.structure.structure_loader import (
    FragmentCachingGlycopeptide, hashable_glycan_glycopeptide_parser,
    HashableGlycanComposition)

glycopeptide = "YPVLN(N-Glycosylation)VTMPN(Deamidation)NGKFDK{Hex:9; HexNAc:2}"


class TestFragmentCachingGlycopeptide(unittest.TestCase):
    def test_mass(self):
        gp = FragmentCachingGlycopeptide(glycopeptide)
        self.assertAlmostEqual(gp.total_mass, 3701.5421769127897, 2)

    def test_parse(self):
        parts = hashable_glycan_glycopeptide_parser(glycopeptide)
        gc = parts[-3]
        self.assertIsInstance(gc, HashableGlycanComposition)


if __name__ == '__main__':
    unittest.main()
