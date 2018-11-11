import operator

import glycopeptidepy
from glycopeptidepy.structure.sequence import (
    _n_glycosylation, _o_glycosylation, _gag_linker_glycosylation)

from glycan_profiling.database import NeutralMassDatabase
from glycan_profiling.database.builder.glycopeptide.common import limiting_combinations

from .core_search import GlycanCombinationRecord


class GlycoformGeneratorBase(object):
    def __init__(self, glycan_combinations, *args, **kwargs):
        self.glycan_combinations = NeutralMassDatabase(
            list(glycan_combinations), operator.attrgetter("dehydrated_mass"))
        super(GlycoformGeneratorBase, self).__init__(*args, **kwargs)

    def handle_glycan_combination(self, peptide_obj, peptide_record, glycan_combination,
                                  glycosylation_sites, core_type):
        glycosylation_sites_unoccupied = set(glycosylation_sites)
        for site in list(glycosylation_sites_unoccupied):
            if peptide_obj[site][1]:
                glycosylation_sites_unoccupied.remove(site)
        site_combinations = list(limiting_combinations(glycosylation_sites_unoccupied, glycan_combination.count))
        result_set = [None for i in site_combinations]
        for i, site_set in enumerate(site_combinations):
            glycoform = peptide_obj.clone()
            glycoform.glycan = glycan_combination.composition.clone()
            for site in site_set:
                glycoform.add_modification(site, core_type.name)
            result_set[i] = glycoform
        return result_set

    def handle_n_glycan(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.n_glycosylation_sites,
            _n_glycosylation)

    def handle_o_glycan(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.o_glycosylation_sites,
            _o_glycosylation)

    def handle_gag_linker(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.gagylation_sites,
            _gag_linker_glycosylation)

    def update_boundary(self, lower, upper):
        self.glycan_combinations = self.glycan_combinations.search_between(lower - 1, upper + 1)
        return self


class PeptideGlycosylator(GlycoformGeneratorBase):
    def __init__(self, peptide_records, glycan_combinations, *args, **kwargs):
        self.peptides = NeutralMassDatabase(peptide_records)
        super(PeptideGlycosylator, self).__init__(glycan_combinations, *args, **kwargs)

    def handle_peptide_mass(self, peptide_mass, intact_mass, error_tolerance=1e-5):
        peptide_records = self.peptides.search_mass_ppm(peptide_mass, error_tolerance)
