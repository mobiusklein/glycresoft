import unittest
import tempfile
import os


from . import fixtures
from .test_constrained_combinatorics import (
    FILE_SOURCE)
from glycresoft.profiler import MzMLGlycanChromatogramAnalyzer, GeneralScorer
from glycresoft.database.builder.glycan import CombinatorialGlycanHypothesisSerializer
from glycresoft.serialize import AnalysisDeserializer


agp_glycomics_mzml = fixtures.get_test_data(
    "AGP_Glycomics_20150930_06.deconvoluted.mzML")


class GlycanProfilerConsumerTest(unittest.TestCase):

    def setup_tempfile(self, content):
        file_name = tempfile.mktemp() + '.tmp'
        open(file_name, 'w').write(content)
        return file_name

    def clear_file(self, path):
        open(path, 'wb').close()

    def _make_hypothesis(self):
        file_name = self.setup_tempfile(FILE_SOURCE)
        builder = CombinatorialGlycanHypothesisSerializer(
            file_name, file_name + '.db')
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
        self.assertEqual(len(gcs), 23)
        self.clear_file(db_file)
        # 'spacing_fit': 0.96367957815527916, 'isotopic_fit': 0.99366937970680247,
        # 'line_score': 0.99780414736388745, 'charge_count': 0.9365769766604084
        self.confirm_score(gcs, "{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:4}", 17.1458)
        # 'spacing_fit': 0.96123524755239487, 'isotopic_fit': 0.97935840584492162,
        # 'line_score': 0.99562733579066764, 'charge_count': 0.7368321292716115
        self.confirm_score(gcs, "{Hex:8; HexNAc:7; Neu5Ac:3}", 13.5279)
        # 'spacing_fit': 0.94565181061625481, 'isotopic_fit': 0.99074210231338733,
        # 'line_score': 0.98925755528448378, 'charge_count': 0.999773289306269
        self.confirm_score(gcs, "{Hex:7; HexNAc:6; Neu5Ac:4}", 20.4438)
        # 'spacing_fit': 0.95567017048597336, 'isotopic_fit': 0.98274665306540443,
        # 'line_score': 0.99706887771172914, 'charge_count': 0.7604540961453831
        self.confirm_score(gcs, "{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:3}", 14.0977)

        ads.close()
        self.clear_file(output_file)

    def test_smoothing_profiler(self):
        db_file = self._make_hypothesis()
        output_file = self.setup_tempfile("")
        task = MzMLGlycanChromatogramAnalyzer(
            db_file, 1, agp_glycomics_mzml, output_file,
            regularize="grid",
            analysis_name="test-analysis",
            scoring_model=GeneralScorer)
        task.start()
        # import cProfile
        # prof = cProfile.Profile()
        # prof.runcall(task.start)
        # prof.print_stats()
        # prof.dump_stats('smooth_profile.pstats')
        self.assertTrue(os.path.exists(output_file))
        ads = AnalysisDeserializer(output_file, analysis_id=1)
        gcs = ads.load_glycan_composition_chromatograms()
        self.assertEqual(len(gcs), 23)
        self.confirm_score(gcs, "{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:4}", 16.1425)
        self.confirm_score(gcs, "{Hex:8; HexNAc:7; Neu5Ac:3}", 8.8510)
        self.confirm_score(gcs, "{Hex:7; HexNAc:6; Neu5Ac:4}", 16.6722)
        network_params = ads.analysis.parameters['network_parameters']
        tau = [0.0, 12.173488161057854, 16.042106463675424, 0.0, 22.061954223206591,
               0.0, 13.928596053020485, 0.0, 9.4348332520855713, 0.0, 0.0, 0.0, 0.0, 0.0]
        for a, b in zip(tau, network_params.tau):
            self.assertAlmostEqual(a, b, 3)
        ads.close()

        self.clear_file(output_file)
        task = MzMLGlycanChromatogramAnalyzer(
            db_file, 1, agp_glycomics_mzml, output_file,
            regularize=0.2,
            regularization_model=network_params,
            analysis_name="test-analysis",
            scoring_model=GeneralScorer)
        task.start()
        ads = AnalysisDeserializer(output_file, analysis_id=1)
        gcs = ads.load_glycan_composition_chromatograms()
        self.assertEqual(len(gcs), 23)
        self.confirm_score(gcs, "{Fuc:1; Hex:7; HexNAc:6; Neu5Ac:4}", 16.7795)
        self.confirm_score(gcs, "{Hex:8; HexNAc:7; Neu5Ac:3}", 10.6734)
        self.confirm_score(gcs, "{Hex:7; HexNAc:6; Neu5Ac:4}", 18.3360)
        self.confirm_score(gcs, "{Fuc:2; Hex:6; HexNAc:5; Neu5Ac:3}", 15.9628)
        network_params = ads.analysis.parameters['network_parameters']
        for a, b in zip(tau, network_params.tau):
            self.assertAlmostEqual(a, b, 3)
        ads.close()
        self.clear_file(output_file)
        self.clear_file(db_file)


if __name__ == '__main__':
    unittest.main()
