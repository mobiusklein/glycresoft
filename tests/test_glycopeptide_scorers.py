import unittest

from glycopeptidepy import Modification
from glypy.structure.glycan_composition import HashableGlycanComposition

from ms_deisotope.output import ProcessedMSFileLoader
from glycresoft.tandem import oxonium_ions

from .fixtures import get_test_data

from glycresoft.tandem.glycopeptide.core_search import GlycanCombinationRecord
from glycresoft.tandem.oxonium_ions import OxoniumIndex
from glycresoft.structure import FragmentCachingGlycopeptide

from glycresoft.tandem.glycopeptide.scoring import (
    base, intensity_scorer, simple_score, binomial_score, coverage_weighted_binomial)

from glycresoft.tandem.peptide.scoring.localize import PTMProphetEvaluator


class TestGlycopeptideScorers(unittest.TestCase):
    def load_spectra(self):
        scan, scan2 = list(ProcessedMSFileLoader(get_test_data("example_glycopeptide_spectra.mzML")))

        return scan, scan2

    def build_structures(self):
        gp = FragmentCachingGlycopeptide(
            'YLGN(N-Glycosylation)ATAIFFLPDEGK{Hex:5; HexNAc:4; Neu5Ac:1}')
        gp2 = FragmentCachingGlycopeptide('YLGN(#:iupac,glycosylation_type=N-Linked:?-?-Hexp-(?-?)-?-?-'
                              'Hexp2NAc-(?-?)-a-D-Manp-(1-6)-[a-D-Neup5Ac-(?-?)-?-?-Hexp-(?-?'
                              ')-?-?-Hexp2NAc-(?-?)-a-D-Manp-(1-3)]b-D-Manp-(1-4)-b-D-Glcp2NA'
                              'c-(1-4)-b-D-Glcp2NAc)ATAIFFLPDEGK')
        return gp, gp2

    def add_oxonium_index(self, scan, gp):
        gc_rec = GlycanCombinationRecord(
            0, 1913.6770236770099, HashableGlycanComposition.parse(gp.glycan_composition), 1, [])
        ox_index = OxoniumIndex()
        ox_index.build_index([gc_rec], all_series=False, allow_ambiguous=False,
                             include_large_glycan_fragments=False,
                             maximum_fragment_size=4)
        index_match = ox_index.match(scan.deconvoluted_peak_set, 2e-5)
        scan.annotations['oxonium_index_match'] = index_match

    def test_simple_coverage_scorer(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = simple_score.SimpleCoverageScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 0.574639463036, 3)
        match = simple_score.SimpleCoverageScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 0.574639463036, 3)

        match = simple_score.SimpleCoverageScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 0.57850568223215082, 3)
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
        self.assertAlmostEqual(match.score, 139.55732008249882, 3)
        match = binomial_score.BinomialSpectrumMatcher.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 191.05060390867396, 3)

    def test_coverage_weighted_binomial(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 103.24070700636717, 3)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 103.24070700636717, 3)

        self.add_oxonium_index(scan, gp)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 103.24070700636717, 3)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 103.24070700636717, 3)

        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 81.04099136734635, 3)
        match = coverage_weighted_binomial.CoverageWeightedBinomialScorer.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 162.35792408346902, 3)

    def test_log_intensity(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = intensity_scorer.LogIntensityScorer.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 55.396555993522334, 3)
        match = intensity_scorer.LogIntensityScorer.evaluate(scan, gp2, rare_signatures=True)
        self.assertAlmostEqual(match.score, 55.396555993522334, 3)

        match = intensity_scorer.LogIntensityScorer.evaluate(scan2, gp)
        self.assertAlmostEqual(match.score, 72.71569538828025, 3)
        match = intensity_scorer.LogIntensityScorer.evaluate(scan2, gp2)
        self.assertAlmostEqual(match.score, 157.97265377375456, 3)

    def test_log_intensity_reweighted(self):
        scan, scan2 = self.load_spectra()
        gp, gp2 = self.build_structures()

        match = intensity_scorer.LogIntensityScorerReweighted.evaluate(scan, gp)
        self.assertAlmostEqual(match.score, 61.839439337876456, 3)
        match = intensity_scorer.LogIntensityScorerReweighted.evaluate(scan, gp2)
        self.assertAlmostEqual(match.score, 61.839439337876456, 3)

        match = intensity_scorer.LogIntensityScorerReweighted.evaluate(
            scan2, gp)
        self.assertAlmostEqual(match.score, 90.76611593053316, 3)
        match = intensity_scorer.LogIntensityScorerReweighted.evaluate(
            scan2, gp2)
        self.assertAlmostEqual(match.score, 149.86761396041246, 3)

    def test_ptm_prophet(self):
        scan, _scan2 = self.load_spectra()
        gp, _gp2 = self.build_structures()

        match = PTMProphetEvaluator(
            scan, gp, modification_rule=Modification("N-Glycosylation").rule,
            respect_specificity=False
        )

        match.score_arrangements()

        match = PTMProphetEvaluator(
            scan, gp, modification_rule=Modification("N-Glycosylation").rule,
            respect_specificity=True
        )

        match.score_arrangements()
