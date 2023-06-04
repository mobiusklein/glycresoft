import unittest

from click.testing import Result, CliRunner

from glycan_profiling import serialize
from glycan_profiling.cli.analyze import search_glycopeptide, search_glycopeptide_multipart

from .fixtures import get_test_data



class _GlycopeptideSearchToolBase:
    mzml_path: str
    classic_result_db: str
    indexed_result_db: str

    classic_search_db: str = get_test_data("agp.db")

    indexed_search_db: str = get_test_data("agp_indexed.db")
    indexed_decoy_search_db: str = get_test_data("agp_indexed_decoy.db")

    def evaluate_classic_search(self, result: Result, output_file: str):
        assert result.exit_code == 0
        db = serialize.DatabaseBoundOperation(output_file)
        ref_db = serialize.DatabaseBoundOperation(self.classic_result_db)

        for model_tp in [serialize.Glycopeptide,
                         serialize.Peptide,
                         serialize.Protein,
                         serialize.GlycanComposition,
                         serialize.GlycopeptideSpectrumSolutionSet,
                         serialize.IdentifiedGlycopeptide]:
            assert db.query(model_tp).count() == ref_db.query(model_tp).count()

    def evaluate_indexed_search(self, result: Result, output_file: str):
        assert result.exit_code == 0
        db = serialize.DatabaseBoundOperation(output_file)
        ref_db = serialize.DatabaseBoundOperation(self.indexed_result_db)

        for model_tp in [serialize.Glycopeptide,
                         serialize.Peptide,
                         serialize.Protein,
                         serialize.GlycanComposition,
                         serialize.GlycopeptideSpectrumMatch,
                         serialize.GlycopeptideSpectrumSolutionSet,
                         serialize.IdentifiedGlycopeptide]:
            assert db.query(model_tp).count() == ref_db.query(model_tp).count()

    def test_classic_search(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as _dir_man:
            result = runner.invoke(search_glycopeptide, [
                "-o", "./classic_agp_search.db",
                self.classic_search_db, self.mzml_path, "1"
            ])
            self.evaluate_classic_search(result, "./classic_agp_search.db")

    def test_indexed_search(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as _dir_man:
            result = runner.invoke(search_glycopeptide_multipart, [
                "-o", "./indexed_agp_search.db",
                "-M",
                self.indexed_search_db,
                self.indexed_decoy_search_db,
                self.mzml_path,
            ])
            self.evaluate_indexed_search(result, "./indexed_agp_search.db")


class TestGlycopeptideSearchTool(_GlycopeptideSearchToolBase, unittest.TestCase):
    mzml_path = get_test_data("20150710_3um_AGP_001_29_30.preprocessed.mzML")
    classic_result_db = get_test_data("classic_agp_search.db")
    indexed_result_db = get_test_data("indexed_agp_search.db")


class TestGlycopeptideSearchToolNoMS2(_GlycopeptideSearchToolBase, unittest.TestCase):
    mzml_path = get_test_data("AGP_Glycomics_20150930_06.deconvoluted.mzML")
    classic_result_db = get_test_data("classic_agp_search_empty.db")
    indexed_result_db = get_test_data("indexed_agp_search_empty.db")
