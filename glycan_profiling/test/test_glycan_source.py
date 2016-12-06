import unittest
import tempfile

from glycan_profiling.database.builder.glycan import glycan_source


FILE_SOURCE = '''
{Hex:3; HexNAc:2}  N-Glycan  O-Glycan
{Fuc:1; Hex:3; HexNAc:2}  N-Glycan
'''


class GlycanSourceTests(unittest.TestCase):

    def setup_tempfile(self):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(FILE_SOURCE)
        return file_name

    def clear_file(self, path):
        open(path, 'wb')

    def test_run(self):
        file_name = self.setup_tempfile()
        builder = glycan_source.TextFileGlycanHypothesisSerializer(
            file_name, file_name + '.db')
        builder.start()
        inst = builder.query(glycan_source.DBGlycanComposition).filter(
            glycan_source.DBGlycanComposition.hypothesis_id == builder.hypothesis_id,
            glycan_source.DBGlycanComposition.composition == "{Hex:3; HexNAc:2}").one()
        self.assertAlmostEqual(inst.calculated_mass, 910.32777, 3)
        self.assertTrue("N-Glycan" in inst.structure_classes)
        builder.engine.dispose()
        self.clear_file(file_name + '.db')
        self.clear_file(file_name)

    def test_run_reduced(self):
        file_name = self.setup_tempfile()
        builder = glycan_source.TextFileGlycanHypothesisSerializer(
            file_name, file_name + '.db', reduction="H2")
        builder.start()
        inst = builder.query(glycan_source.DBGlycanComposition).filter(
            glycan_source.DBGlycanComposition.hypothesis_id == builder.hypothesis_id,
            glycan_source.DBGlycanComposition.composition == "{Hex:3; HexNAc:2}$H2").one()
        self.assertAlmostEqual(inst.calculated_mass, 912.3434, 3)
        builder.engine.dispose()
        self.clear_file(file_name + '.db')
        self.clear_file(file_name)

    def test_run_permethylated(self):
        file_name = self.setup_tempfile()
        builder = glycan_source.TextFileGlycanHypothesisSerializer(
            file_name, file_name + '.db', reduction="H2", derivatization='methyl')
        builder.start()
        inst = builder.query(glycan_source.DBGlycanComposition).filter(
            glycan_source.DBGlycanComposition.hypothesis_id == builder.hypothesis_id,
            glycan_source.DBGlycanComposition.composition == "{Hex^Me:3; HexNAc^Me:2}$C1H4").one()
        self.assertAlmostEqual(inst.calculated_mass, 1164.6251311968801, 3)
        builder.engine.dispose()
        self.clear_file(file_name + '.db')
        self.clear_file(file_name)


if __name__ == '__main__':
    unittest.main()
