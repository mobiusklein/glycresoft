import unittest

import glypy
from glycopeptidepy import PeptideSequence
from ms_deisotope.output import ProcessedMSFileLoader

from glycresoft.tandem.glycopeptide import core_search
from glycresoft.tandem.glycopeptide.core_search import (
    GlycanCombinationRecord, GlycanTypes, GlycanFilteringPeptideMassEstimator)

from .fixtures import get_test_data


peptide_mass = PeptideSequence("YLGNATAIFFLPDEGK").mass
gc1 = glypy.glycan_composition.HashableGlycanComposition.parse("{Hex:5; HexNAc:4; Neu5Ac:1}")
gc2 = glypy.glycan_composition.HashableGlycanComposition.parse("{Hex:5; HexNAc:4; Neu5Ac:2}")
gc3 = glypy.glycan_composition.HashableGlycanComposition.parse("{Hex:6; HexNAc:5; Neu5Ac:2}")
glycan_compositions = [gc1, gc2, gc3]

glycan_database = []
for i, gc in enumerate(glycan_compositions):
    record = GlycanCombinationRecord(
        i + 1, gc.mass() - gc.composition_offset.mass, gc, 1, [
            GlycanTypes.n_glycan,
            GlycanTypes.o_glycan,

        ])
    glycan_database.append(record)



class TestGlycanFilteringPeptideMassEstimator(unittest.TestCase):
    def load_spectra(self):
        return list(ProcessedMSFileLoader(get_test_data("example_glycopeptide_spectra.mzML")))

    def make_estimator(self):
        return GlycanFilteringPeptideMassEstimator(glycan_database)

    def test_estimate(self):
        estimator = self.make_estimator()
        scans = self.load_spectra()
        scan = scans[0]
        ranked = estimator.match(scan)
        match = ranked[0]
        print(match)
        self.assertAlmostEqual(match.score, 29.715553766294754, 3)


if __name__ == "__main__":
    unittest.main()
