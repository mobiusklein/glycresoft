import unittest
import tempfile


from glycresoft import serialize

from glycresoft.database.prebuilt import biosynthesis_human_n_linked


class GlycanCombinatoricsTests(unittest.TestCase):
    def setup_tempfile(self):
        file_name = tempfile.mktemp() + '.tmp'
        open(file_name, 'w').write('')
        return file_name

    def clear_file(self, path):
        open(path, 'wb')

    def setup_compositions(self, database_path, **kwargs):
        builder = biosynthesis_human_n_linked.BiosynthesisHumanNGlycansBuilder()
        result = builder.build(database_path, **kwargs)
        return result

    def test_prebuilt_native(self):
        path = self.setup_tempfile()
        hypothesis_builder = self.setup_compositions(path)
        glycan_count = hypothesis_builder.query(serialize.GlycanComposition).count()
        self.assertEqual(448, glycan_count)
        hypothesis_builder.close()
        self.clear_file(path)

    def test_prebuilt_permethylated(self):
        path = self.setup_tempfile()
        hypothesis_builder = self.setup_compositions(path, derivatization='methyl')
        glycan_count = hypothesis_builder.query(serialize.GlycanComposition).count()
        self.assertEqual(448, glycan_count)
        hypothesis_builder.close()
        self.clear_file(path)


if __name__ == '__main__':
    unittest.main()
