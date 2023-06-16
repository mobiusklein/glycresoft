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
