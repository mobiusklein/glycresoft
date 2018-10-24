import unittest

from glycopeptidepy import PeptideSequence
from ms_deisotope.output import ProcessedMzMLDeserializer

from glycan_profiling.test.fixtures import get_test_data

from glycan_profiling.tandem.glycopeptide.scoring import (
    base, intensity_scorer, simple_score, binomial_score, coverage_weighted_binomial)


class TestGlycopeptideScorers(unittest.TestCase):
    def load_spectra(self):
        return list(ProcessedMzMLDeserializer(get_test_data("example_glycopeptide_spectra.mzML")))

    def build_structures(self):
        gp = PeptideSequence('YLGN(N-Glycosylation)ATAIFFLPDEGK{Hex:5; HexNAc:4; Neu5Ac:1}')
        gp2 = PeptideSequence('YLGN(#:iupac,glycosylation_type=N-Linked:?-?-Hexp-(?-?)-?-?-'
                              'Hexp2NAc-(?-?)-a-D-Manp-(1-6)-[a-D-Neup5Ac-(?-?)-?-?-Hexp-(?-?'
                              ')-?-?-Hexp2NAc-(?-?)-a-D-Manp-(1-3)]b-D-Manp-(1-4)-b-D-Glcp2NA'
                              'c-(1-4)-b-D-Glcp2NAc)ATAIFFLPDEGK')
        return gp, gp2

    def test_simple_coverage_scorer(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = simple_score.SimpleCoverageScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 0.574639463036, 3)
        match = simple_score.SimpleCoverageScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 0.574639463036, 3)

        match = simple_score.SimpleCoverageScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 0.461839015565, 3)
        match = simple_score.SimpleCoverageScorer.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 0.848213154345, 3)

    def test_binomial_scorer(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = binomial_score.BinomialSpectrumMatcher.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 179.12869707912699, 3)
        match = binomial_score.BinomialSpectrumMatcher.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 179.12869707912699, 3)

        match = binomial_score.BinomialSpectrumMatcher.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 119.31335056352407, 3)
        match = binomial_score.BinomialSpectrumMatcher.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 191.42842627069271, 3)

    def test_coverage_weighted_binomial(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 103.248862672, 3)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 103.248862672, 3)

        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 55.4180047361, 3)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 162.686553646, 3)

    def test_log_intensity(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = intensity_scorer.LogIntensityScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 129.8531117, 3)
        match = intensity_scorer.LogIntensityScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 129.8531117, 3)

        match = intensity_scorer.LogIntensityScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 94.1181520719, 3)
        match = intensity_scorer.LogIntensityScorer.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 262.759675688, 3)
