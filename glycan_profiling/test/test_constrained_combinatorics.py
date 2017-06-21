import unittest
import tempfile

from io import StringIO

from glycan_profiling.database.builder.glycan import constrained_combinatorics


FILE_SOURCE = '''Hex 3 12
HexNAc 2 10
Fuc 0 5
Neu5Ac 0 4

Fuc < HexNAc
HexNAc > NeuAc + 1
'''      # The mismatch between Neu5Ac <-> NeuAc tests the IUPAC normalization mechanism


FILE_SOURCE2 = '''Hex 3 12
HexNAc 2 10
Fuc 0 5
Neu5Ac 0 4

Fuc < HexNAc
HexNAc - 1 > NeuAc
'''

FILE_SOURCE3 = '''Hex 3 12
HexNAc 2 10
Fuc 0 5
Neu5Ac 0 4

Fuc < HexNAc
NeuAc < HexNAc - 1
'''

FILE_SOURCE4 = '''Hex 3 12
HexNAc 2 10
Fuc 0 5
Neu5Ac 0 4

Fuc < HexNAc
NeuAc < HexNAc
'''


class GlycanCombinatoricsTests(unittest.TestCase):

    def setup_tempfile(self):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(FILE_SOURCE)
        return file_name

    def clear_file(self, path):
        open(path, 'wb').close()

    def test_run(self):
        file_name = self.setup_tempfile()
        builder = constrained_combinatorics.CombinatorialGlycanHypothesisSerializer(
            file_name, file_name + '.db')
        builder.run()
        inst = builder.query(constrained_combinatorics.DBGlycanComposition).filter(
            constrained_combinatorics.DBGlycanComposition.hypothesis_id == builder.hypothesis_id,
            constrained_combinatorics.DBGlycanComposition.composition == "{Hex:3; HexNAc:2}").one()
        self.assertAlmostEqual(inst.calculated_mass, 910.32777, 3)

        self.do_neuac_check(builder)

        builder.engine.dispose()
        self.clear_file(file_name + '.db')
        self.clear_file(file_name)

    def do_neuac_check(self, builder):
        i = 0
        for composition in builder.query(constrained_combinatorics.DBGlycanComposition):
            composition = composition.convert()
            i += 1
            self.assertGreater(composition["HexNAc"] - 1, composition['Neu5Ac'])


class GlycanCombinatoricsTestsWithReversedTermInConditional(GlycanCombinatoricsTests):

    def setup_tempfile(self):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(FILE_SOURCE2)
        return file_name


class GlycanCombinatoricsTestsWithReversedConditional(GlycanCombinatoricsTests):

    def setup_tempfile(self):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(FILE_SOURCE3)
        return file_name


class GlycanCombinatoricsTestsWithoutSubtractionInConditional(GlycanCombinatoricsTests):

    def setup_tempfile(self):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(FILE_SOURCE4)
        return file_name

    def do_neuac_check(self, builder):
        self.assertRaises(AssertionError, lambda: super(
            GlycanCombinatoricsTestsWithoutSubtractionInConditional, self).do_neuac_check(builder))


if __name__ == '__main__':
    unittest.main()
