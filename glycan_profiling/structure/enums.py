from glypy.utils import Enum


class SpectrumMatchClassification(Enum):
    target_peptide_target_glycan = 0
    target_peptide_decoy_glycan = 1
    decoy_peptide_target_glycan = 2
    decoy_peptide_decoy_glycan = decoy_peptide_target_glycan | target_peptide_decoy_glycan
