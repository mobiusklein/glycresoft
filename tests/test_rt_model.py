import unittest

import glycopeptidepy

from glypy.structure.glycan_composition import HashableGlycanComposition

from glycresoft.chromatogram_tree.mass_shift import Unmodified, Ammonium
from glycresoft.scoring.elution_time_grouping.structure import ChromatogramProxy, GlycopeptideChromatogramProxy
from glycresoft.scoring.elution_time_grouping.reviser import IsotopeRule


class TestChromatogramProxy(unittest.TestCase):
    def _create_instance(self) -> ChromatogramProxy:
        gc = HashableGlycanComposition(Hex=5, HexNAc=4, NeuAc=2, Fuc=1)
        return ChromatogramProxy(gc.mass(), 25.0, 1e6, gc, None, [Unmodified, Ammonium])

    def test_apex_time(self):
        proxy = self._create_instance()
        self.assertAlmostEqual(proxy.apex_time, 25.0, 3)

    def test_weight(self):
        proxy = self._create_instance()
        assert proxy.weight == 1.0

    def test_copy(self):
        proxy = self._create_instance()
        assert proxy == proxy.copy()

    def test_shift_glycan(self):
        proxy = self._create_instance()
        new = proxy.shift_glycan_composition(HashableGlycanComposition(Fuc=2, NeuAc=-1))
        assert new.glycan_composition == HashableGlycanComposition(
            Hex=5, HexNAc=4, NeuAc=1, Fuc=3)


class TestGlycopeptideChromatogramProxy(TestChromatogramProxy):
    def _create_instance(self) -> GlycopeptideChromatogramProxy:
        gc = HashableGlycanComposition(Hex=5, HexNAc=4, NeuAc=2, Fuc=1)
        glycopeptide = glycopeptidepy.parse("NEEYN(N-Glycosylation)K" + str(gc))
        inst = GlycopeptideChromatogramProxy(glycopeptide.total_mass, 25.0, 1e6, gc, None, [Unmodified, Ammonium], structure=glycopeptide)
        assert inst.structure == glycopeptide
        return inst

    def test_shift_glycan(self):
        proxy = self._create_instance()
        new = proxy.shift_glycan_composition(
            HashableGlycanComposition(Fuc=2, NeuAc=-1))
        assert new.glycan_composition == HashableGlycanComposition(
            Hex=5, HexNAc=4, NeuAc=1, Fuc=3)
        assert new.structure == glycopeptidepy.parse(
            "NEEYN(N-Glycosylation)K" + str(new.glycan_composition))
        assert new.kwargs['original_structure'] == str(proxy.structure)


class TestRevisionRule(unittest.TestCase):
    def _create_instance(self) -> GlycopeptideChromatogramProxy:
        gc = HashableGlycanComposition(Hex=5, HexNAc=4, NeuAc=1, Fuc=3)
        glycopeptide = glycopeptidepy.parse("NEEYN(N-Glycosylation)K" + str(gc))
        inst = GlycopeptideChromatogramProxy(glycopeptide.total_mass, 25.0, 1e6, gc, None, [Unmodified, Ammonium], structure=glycopeptide)
        assert inst.structure == glycopeptide
        return inst

    def test_apply(self):
        proxy = self._create_instance()
        rule = IsotopeRule.clone()
        new = rule(proxy)
        assert new.glycan_composition == HashableGlycanComposition(
            Hex=5, HexNAc=4, NeuAc=2, Fuc=1)
        assert new.structure == glycopeptidepy.parse(
            "NEEYN(N-Glycosylation)K" + str(new.glycan_composition))
        assert new.kwargs['original_structure'] == str(proxy.structure)

    def test_apply_with_cache(self):
        proxy = self._create_instance()
        rule = IsotopeRule.with_cache()
        new = rule(proxy)
        assert new.glycan_composition == HashableGlycanComposition(
            Hex=5, HexNAc=4, NeuAc=2, Fuc=1)
        assert new.structure == glycopeptidepy.parse(
            "NEEYN(N-Glycosylation)K" + str(new.glycan_composition))
        assert new.kwargs['original_structure'] == str(proxy.structure)

        new2 = rule(proxy)
        assert new2.glycan_composition == HashableGlycanComposition(
            Hex=5, HexNAc=4, NeuAc=2, Fuc=1)
        assert new2.structure == glycopeptidepy.parse(
            "NEEYN(N-Glycosylation)K" + str(new.glycan_composition))
        assert new2.kwargs['original_structure'] == str(proxy.structure)

        assert new2 is not new
        assert new2.structure is not new.structure
