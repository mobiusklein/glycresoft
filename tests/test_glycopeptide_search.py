import unittest
import traceback

from lxml import html
from click.testing import Result, CliRunner

from glycresoft import serialize
from glycresoft.cli.analyze import (
    search_glycopeptide,
    search_glycopeptide_multipart,
)

from .fixtures import get_test_data


class _GlycopeptideSearchToolBase:
    mzml_path: str
    classic_result_db: str
    indexed_result_db: str

    classic_search_db: str = get_test_data("agp.db")

    indexed_search_db: str = get_test_data("agp_indexed.db")
    indexed_decoy_search_db: str = get_test_data("agp_indexed_decoy.db")

    indexed_gp_csv_headers = [
        "glycopeptide",
        "analysis",
        "neutral_mass",
        "mass_accuracy",
        "ms1_score",
        "ms2_score",
        "q_value",
        "total_signal",
        "start_time",
        "end_time",
        "apex_time",
        "charge_states",
        "msms_count",
        "peptide_start",
        "peptide_end",
        "protein_name",
        "mass_shifts",
        "predicted_apex_interval_start",
        "predicted_apex_interval_end",
        "retention_time_score",
        "group_id",
        "glycopeptide_score",
        "peptide_score",
        "glycan_score",
        "glycan_coverage",
        "total_q_value",
        "peptide_q_value",
        "glycan_q_value",
        "glycopeptide_q_value",
        "localizations",
        "n_glycosylation_sites",

    ]

    indexed_gpsm_csv_header = [
        "glycopeptide",
        "analysis",
        "neutral_mass",
        "mass_accuracy",
        "mass_shift_name",
        "scan_id",
        "scan_time",
        "charge",
        "ms2_score",
        "q_value",
        "precursor_abundance",
        "peptide_start",
        "peptide_end",
        "protein_name",
        "is_best_match",
        "is_precursor_fit",
        "rank",
        "group_id",
        "peptide_score",
        "glycan_score",
        "glycan_coverage",
        "peptide_q_value",
        "glycan_q_value",
        "glycopeptide_q_value",
        "localizations",
        "n_glycosylation_sites",
    ]

    gp_csv_suffix = "-glycopeptides.csv"
    gpsm_csv_suffix = "-glycopeptide-spectrum-matches.csv"
    html_report_suffix = "-report.html"

    def parse_html_report(self, html_path: str):
        tree = html.parse(html_path)
        n_prots = len(tree.findall(
            ".//section[@id='protein-table-container']/table/tbody/tr"))
        n_gps = len(tree.xpath(
            ".//*[@class and contains(concat(' ', normalize-space(@class), ' '), ' glycopeptide-detail-table-row ')]"))
        return n_prots, n_gps

    def evaluate_classic_search(self, result: Result, output_file: str):
        if result.exit_code != 0:
            print(f"Exit Code {result.exit_code}")
            print(f"Exit Code {traceback.format_exception(*result.exc_info)}")
            print(result.output)
        assert result.exit_code == 0
        db = serialize.DatabaseBoundOperation(output_file)
        ref_db = serialize.DatabaseBoundOperation(self.classic_result_db)

        for model_tp in [
            serialize.Glycopeptide,
            serialize.Peptide,
            serialize.Protein,
            serialize.GlycanComposition,
            serialize.GlycopeptideSpectrumSolutionSet,
            serialize.IdentifiedGlycopeptide,
        ]:
            assert db.query(model_tp).count() == ref_db.query(model_tp).count()

        prefix = output_file.rsplit(".", 1)[0]

        html_path = prefix + self.html_report_suffix
        n_prots, n_gps = self.parse_html_report(html_path)

        id_n_prots = db.query(serialize.Protein).join(
            serialize.Glycopeptide).join(
            serialize.IdentifiedGlycopeptide).group_by(
            serialize.Protein.id).count()
        assert n_prots == id_n_prots

        id_n_gps = db.query(serialize.IdentifiedGlycopeptide).count()
        assert id_n_gps == n_gps

    def evaluate_indexed_search(self, result: Result, output_file: str):
        if result.exit_code != 0:
            print(f"Exit Code {result.exit_code}")
            print(f"Exit Code {traceback.format_exception(*result.exc_info)}")
            print(result.output)
        assert result.exit_code == 0
        db = serialize.DatabaseBoundOperation(output_file)
        ref_db = serialize.DatabaseBoundOperation(self.indexed_result_db)

        for model_tp in [
            serialize.Glycopeptide,
            serialize.Peptide,
            serialize.Protein,
            serialize.GlycanComposition,
            serialize.GlycopeptideSpectrumMatch,
            serialize.GlycopeptideSpectrumSolutionSet,
            serialize.IdentifiedGlycopeptide,
        ]:
            assert db.query(model_tp).count() == ref_db.query(model_tp).count()

        prefix = output_file.rsplit(".", 1)[0]

        html_path = prefix + self.html_report_suffix
        n_prots, n_gps = self.parse_html_report(html_path)

        id_n_prots = db.query(serialize.Protein).join(
            serialize.Glycopeptide).join(
            serialize.IdentifiedGlycopeptide).group_by(
            serialize.Protein.id).count()
        assert n_prots == id_n_prots

        id_n_gps = db.query(serialize.IdentifiedGlycopeptide).count()
        assert id_n_gps == n_gps

        with open(prefix + self.gp_csv_suffix, 'rt') as fh:
            headers = fh.readline().strip().split(",")
            assert set(headers) <= set(self.indexed_gp_csv_headers)

        with open(prefix + self.gpsm_csv_suffix, 'rt') as fh:
            headers = fh.readline().strip().split(",")
            assert set(headers) <= set(self.indexed_gpsm_csv_header)

    def test_classic_search(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as _dir_man:
            result = runner.invoke(
                search_glycopeptide,
                [
                    "-o",
                    "./classic_agp_search.db",
                    self.classic_search_db,
                    self.mzml_path,
                    "1",
                    "--export",
                    "csv",
                    "--export",
                    "html",
                    "--export",
                    "psm-csv",
                ],
            )
            self.evaluate_classic_search(result, "./classic_agp_search.db")

    def test_indexed_search(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as _dir_man:
            result = runner.invoke(
                search_glycopeptide_multipart,
                [
                    "-o",
                    "./indexed_agp_search.db",
                    "-M",
                    "--export",
                    "csv",
                    "--export",
                    "html",
                    "--export",
                    "psm-csv",
                    self.indexed_search_db,
                    self.indexed_decoy_search_db,
                    self.mzml_path,
                ],
            )
            self.evaluate_indexed_search(result, "./indexed_agp_search.db")


class TestGlycopeptideSearchTool(_GlycopeptideSearchToolBase, unittest.TestCase):
    mzml_path = get_test_data("20150710_3um_AGP_001_29_30.preprocessed.mzML")
    classic_result_db = get_test_data("classic_agp_search.db")
    indexed_result_db = get_test_data("indexed_agp_search.db")


class TestGlycopeptideSearchToolNoMS2(_GlycopeptideSearchToolBase, unittest.TestCase):
    mzml_path = get_test_data("AGP_Glycomics_20150930_06.deconvoluted.mzML")
    classic_result_db = get_test_data("classic_agp_search_empty.db")
    indexed_result_db = get_test_data("indexed_agp_search_empty.db")
