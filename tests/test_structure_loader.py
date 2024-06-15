import unittest

from glycresoft.structure.structure_loader import (
    FragmentCachingGlycopeptide, hashable_glycan_glycopeptide_parser,
    HashableGlycanComposition, GlycanCompositionWithOffsetProxy)

from glycopeptidepy.test.sequence_test_suite import PeptideSequenceSuiteBase

glycopeptide = "YPVLN(N-Glycosylation)VTMPN(Deamidation)NGKFDK{Hex:9; HexNAc:2}"


class TestFragmentCachingGlycopeptide(PeptideSequenceSuiteBase, unittest.TestCase):
    def parse_sequence(self, seqstr):
        return FragmentCachingGlycopeptide(seqstr)

    def test_mass(self):
        gp = FragmentCachingGlycopeptide(glycopeptide)
        self.assertAlmostEqual(gp.total_mass, 3701.5421769127897, 2)

    def test_parse(self):
        parts = hashable_glycan_glycopeptide_parser(glycopeptide)
        gc = parts[-3]
        self.assertIsInstance(
            gc, (HashableGlycanComposition, GlycanCompositionWithOffsetProxy))

    test_cad_fragmentation = None
    test_glycan_fragments_stubs = None


if __name__ == '__main__':
    unittest.main()
