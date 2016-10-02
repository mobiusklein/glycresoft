import unittest
import tempfile

from glycan_profiling.database.builder.glycan import glycan_source


FILE_SOURCE = '''
{Hex:3; HexNAc:2}
{Fuc:1; Hex:3; HexNAc:2}
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
        builder.run()
        inst = builder.query(glycan_source.DBGlycanComposition).filter(
            glycan_source.DBGlycanComposition.hypothesis_id == builder.hypothesis_id,
            glycan_source.DBGlycanComposition.composition == "{Hex:3; HexNAc:2}").one()
        self.assertAlmostEqual(inst.calculated_mass, 910.32777, 3)
        builder.engine.dispose()
        self.clear_file(file_name + '.db')
        self.clear_file(file_name)


if __name__ == '__main__':
    unittest.main()
