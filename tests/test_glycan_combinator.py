import unittest
import tempfile

from glycresoft.database.builder.glycan import glycan_combinator, glycan_source
from glycresoft.database.builder.glycopeptide.common import GlycopeptideHypothesisSerializerBase


class GlycanCombinatoricsTests(unittest.TestCase):
    def setup_tempfile(self):
        file_name = tempfile.mktemp() + '.tmp'
        open(file_name, 'w').write(FILE_SOURCE)
        return file_name

    def clear_file(self, path):
        open(path, 'wb')

    def setup_compositions(self, source_file, database_path):
        builder = glycan_source.TextFileGlycanHypothesisSerializer(source_file, database_path)
        builder.run()
        return builder

    def test_combinator(self):
        source_file = self.setup_tempfile()
        database_path = source_file + '.db'
        builder = self.setup_compositions(source_file, database_path)
        self.assertTrue(builder.query(glycan_combinator.GlycanComposition).count() > 0)

        # Create the glycopeptide hypothesis that GlycanCombination objects must be associated with
        glycopeptide_builder_stub = GlycopeptideHypothesisSerializerBase(database_path, "test", builder.hypothesis_id)

        combinator = glycan_combinator.GlycanCombinationSerializer(
            database_path, builder.hypothesis_id, glycopeptide_builder_stub.hypothesis.id, max_size=2)
        combinator.run()
        inst = builder.query(glycan_combinator.GlycanCombination).filter(
            glycan_combinator.GlycanCombination.hypothesis_id == builder.hypothesis_id,
            glycan_combinator.GlycanCombination.count == 1,
            glycan_combinator.GlycanCombination.composition == "{Hex:5; HexNAc:4; Neu5Ac:2}").one()
        self.assertAlmostEqual(inst.calculated_mass, 2222.7830048, 5)

        inst = builder.query(glycan_combinator.GlycanCombination).filter(
            glycan_combinator.GlycanCombination.hypothesis_id == builder.hypothesis_id,
            glycan_combinator.GlycanCombination.count == 2,
            glycan_combinator.GlycanCombination.composition == "{Hex:10; HexNAc:8; Neu5Ac:3}").first()

        self.assertAlmostEqual(inst.calculated_mass, 4154.47059322789, 5)

        self.clear_file(source_file)
        self.clear_file(source_file + '.db')


FILE_SOURCE = '''
{Hex:5; HexNAc:4; Neu5Ac:1}
{Hex:5; HexNAc:4; Neu5Ac:2}
{Fuc:1; Hex:5; HexNAc:4; Neu5Ac:2}
{Hex:6; HexNAc:4; Neu5Ac:2}
{Fuc:3; Hex:8; HexNAc:4}
{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:1}
{Fuc:1; Hex:6; HexNAc:5; Neu5Ac:2}
{Fuc:3; Hex:6; HexNAc:5; Neu5Ac:1}
{Hex:6; HexNAc:5; Neu5Ac:3}
{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:2}
{Hex:7; HexNAc:6; Neu5Ac:2}
{Fuc:1; Hex:6; HexNAc:5; Neu5Ac:3}
{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:2}
{Hex:10; HexNAc:6; Neu5Ac:1}
{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:3}
{Hex:7; HexNAc:6; Neu5Ac:3}
{Fuc:2; Hex:7; HexNAc:6; Neu5Ac:2}
{Hex:8; HexNAc:7; Neu5Ac:2}
{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:3}
{Hex:7; HexNAc:6; Neu5Ac:4}
{Fuc:2; Hex:7; HexNAc:6; Neu5Ac:3}
{Hex:8; HexNAc:7; Neu5Ac:3}
{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:4}
{Fuc:1; Hex:8; HexNAc:7; Neu5Ac:3}
{Fuc:2; Hex:7; HexNAc:6; Neu5Ac:4}
{Hex:8; HexNAc:7; Neu5Ac:4}
{Fuc:2; Hex:8; HexNAc:7; Neu5Ac:3}
{Fuc:2; Hex:8; HexNAc:10; Neu5Ac:1}
{Fuc:3; Hex:7; HexNAc:6; Neu5Ac:4}
{Hex:9; HexNAc:8; Neu5Ac:3}
{Fuc:5; Hex:7; HexNAc:9; Neu5Ac:1}
{Fuc:1; Hex:8; HexNAc:7; Neu5Ac:4}
{Fuc:2; Hex:8; HexNAc:7; Neu5Ac:4}
{Hex:9; HexNAc:8; Neu5Ac:4}
{Fuc:2; Hex:9; HexNAc:8; Neu5Ac:3}
{Fuc:3; Hex:8; HexNAc:7; Neu5Ac:4}
{Fuc:1; Hex:9; HexNAc:8; Neu5Ac:4}
{Fuc:3; Hex:11; HexNAc:8; Neu5Ac:2}
{Hex:9; HexNAc:8; Neu5Ac:5}
{Fuc:2; Hex:9; HexNAc:8; Neu5Ac:4}
{Fuc:3; Hex:9; HexNAc:8; Neu5Ac:4}
{Fuc:2; Hex:9; HexNAc:8; Neu5Ac:5}
{Fuc:5; Hex:12; HexNAc:9; Neu5Ac:2}
'''

if __name__ == '__main__':
    unittest.main()
