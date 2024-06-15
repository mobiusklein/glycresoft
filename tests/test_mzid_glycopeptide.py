import unittest
import tempfile

from glycresoft.serialize.hypothesis.peptide import Peptide, Protein, Glycopeptide
from glycresoft.database.builder.glycopeptide import informed_glycopeptide
from glycresoft.database.builder.glycan import (
    CombinatorialGlycanHypothesisSerializer)
from glycresoft import serialize

from . import fixtures
from .test_constrained_combinatorics import FILE_SOURCE as GLYCAN_RULE_FILE_SOURCE


MZID_PATH = fixtures.get_test_data("AGP_Proteomics2.mzid")


class MzIdGlycopeptideTests(unittest.TestCase):

    def setup_tempfile(self, source):
        file_name = tempfile.mktemp() + '.tmp'
        open(file_name, 'w').write(source)
        return file_name

    def clear_file(self, path):
        open(path, 'wb')

    def test_build_hypothesis(self):
        glycan_file = self.setup_tempfile(GLYCAN_RULE_FILE_SOURCE)
        mzid_path = MZID_PATH
        db_file = glycan_file + '.db'

        glycan_builder = CombinatorialGlycanHypothesisSerializer(glycan_file, db_file)
        glycan_builder.run()

        glycopeptide_builder = informed_glycopeptide.MultipleProcessMzIdentMLGlycopeptideHypothesisSerializer(
            mzid_path, db_file, glycan_builder.hypothesis_id)
        glycopeptide_builder.run()

        gp_count = glycopeptide_builder.query(Glycopeptide).count()
        with_uniprot = 769500
        with_uniprot_without_variable_signal_peptide = 659300
        without_uniprot = 651700
        without_any_external = 646000
        self.assertIn(
            gp_count, (with_uniprot, without_uniprot, without_any_external))

        redundancy = glycopeptide_builder.query(
            Glycopeptide.glycopeptide_sequence,
            Protein.name,
            serialize.func.count(Glycopeptide.glycopeptide_sequence)).join(
            Glycopeptide.protein).join(Glycopeptide.peptide).group_by(
                Glycopeptide.glycopeptide_sequence,
                Protein.name,
                Peptide.start_position,
                Peptide.end_position).yield_per(1000)

        for sequence, protein, count in redundancy:
            self.assertEqual(count, 1, "%s in %s has multiplicity %d" % (sequence, protein, count))

        for case in glycopeptide_builder.query(Glycopeptide).filter(
                Glycopeptide.glycopeptide_sequence ==
                "SVQEIQATFFYFTPN(N-Glycosylation)K{Hex:5; HexNAc:4; Neu5Ac:2}").all():
            self.assertAlmostEqual(case.calculated_mass, 4123.718954557139, 5)

        self.clear_file(glycan_file)
        self.clear_file(db_file)


if __name__ == '__main__':
    unittest.main()
