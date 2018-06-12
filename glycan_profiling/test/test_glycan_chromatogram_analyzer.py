import unittest
import tempfile
import os


from glycan_profiling.test import fixtures
from glycan_profiling.test.test_constrained_combinatorics import (
    FILE_SOURCE)
from glycan_profiling.profiler import MzMLGlycanChromatogramAnalyzer, GeneralScorer
from glycan_profiling.database.builder.glycan import CombinatorialGlycanHypothesisSerializer
from glycan_profiling.serialize import AnalysisDeserializer


agp_glycomics_mzml = fixtures.get_test_data("AGP_Glycomics_20150930_06.deconvoluted.mzML")


class GlycanProfilerConsumerTest(unittest.TestCase):
    def setup_tempfile(self, content):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(content)
        return file_name

    def clear_file(self, path):
        open(path, 'wb').close()

    def _make_hypothesis(self):
        file_name = self.setup_tempfile(FILE_SOURCE)
        builder = CombinatorialGlycanHypothesisSerializer(file_name, file_name + '.db')
        builder.run()
        self.clear_file(file_name)
        return file_name + '.db'

    def confirm_score(self, chroma, key, score):
        match = chroma.find_key(key)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(score, match.score, 1)

    def confirm_absent(self, chroma, key):
        match = chroma.find_key(key)
        self.assertIsNone(match)

    def test_profiler(self):
        db_file = self._make_hypothesis()
        output_file = self.setup_tempfile("")
        task = MzMLGlycanChromatogramAnalyzer(
            db_file, 1, agp_glycomics_mzml, output_file,
            analysis_name="test-analysis",
            scoring_model=GeneralScorer)
        task.start()
        self.assertTrue(os.path.exists(output_file))
        ads = AnalysisDeserializer(output_file)
        gcs = ads.load_glycan_composition_chromatograms()
        self.clear_file(db_file)
        # 'spacing_fit': 0.96367957815527916, 'isotopic_fit': 0.99366937970680247,
        # 'line_score': 0.99780414736388745, 'charge_count': 0.9365769766604084
        self.confirm_score(gcs, "{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:4}", 17.1458)
        # 'spacing_fit': 0.96123524755239487, 'isotopic_fit': 0.97935840584492162,
        # 'line_score': 0.99562733579066764, 'charge_count': 0.7368321292716115
        self.confirm_score(gcs, "{Hex:8; HexNAc:7; Neu5Ac:3}", 13.5279)
        # 'spacing_fit': 0.94565181061625481, 'isotopic_fit': 0.99074210231338733,
        # 'line_score': 0.98863881600507719, 'charge_count': 0.999773289306269
        self.confirm_score(gcs, "{Hex:7; HexNAc:6; Neu5Ac:4}", 20.3872)
        # 'spacing_fit': 0.95567017048597336, 'isotopic_fit': 0.98274665306540443,
        # 'line_score': 0.99640424549974071, 'charge_count': 0.7604540961453831
        self.confirm_score(gcs, "{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:3}", 13.8927)

        ads.close()
        self.clear_file(output_file)


if __name__ == '__main__':
    unittest.main()
