from glypy.utils import Enum


class SpectrumMatchClassification(Enum):
    target_peptide_target_glycan = 0
    target_peptide_decoy_glycan = 1
    decoy_peptide_target_glycan = 2
    decoy_peptide_decoy_glycan = decoy_peptide_target_glycan | target_peptide_decoy_glycan


class AnalysisTypeEnum(Enum):
    glycopeptide_lc_msms = 0
    glycan_lc_ms = 1


class GlycopeptideSearchStrategy(Enum):
    target_internal_decoy_competition = "target-internal-decoy-competition"
    target_decoy_competition = "target-decoy-competition"
    multipart_target_decoy_competition = "multipart-target-decoy-competition"


AnalysisTypeEnum.glycopeptide_lc_msms.add_name("Glycopeptide LC-MS/MS")
AnalysisTypeEnum.glycan_lc_ms.add_name("Glycan LC-MS")


class GlycopeptideFDREstimationStrategy(Enum):
    multipart_gamma_gaussian_mixture = 0
    peptide_fdr = 1
    glycan_fdr = 2
    glycopeptide_fdr = 3
    peptide_or_glycan = 4


GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture.add_name(
    "multipart")
GlycopeptideFDREstimationStrategy.multipart_gamma_gaussian_mixture.add_name(
    "joint")
GlycopeptideFDREstimationStrategy.peptide_fdr.add_name("peptide")
GlycopeptideFDREstimationStrategy.glycan_fdr.add_name("glycan")
GlycopeptideFDREstimationStrategy.peptide_or_glycan.add_name('any')
