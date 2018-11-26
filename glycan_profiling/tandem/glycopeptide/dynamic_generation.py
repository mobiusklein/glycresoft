import operator

from glycopeptidepy import GlycosylationType
from glycopeptidepy.structure.sequence import (
    _n_glycosylation, _o_glycosylation, _gag_linker_glycosylation)

from glycan_profiling.database.mass_collection import NeutralMassDatabase, SearchableMassCollection
from glycan_profiling.database.builder.glycopeptide.common import limiting_combinations

from .core_search import GlycanCombinationRecord


class GlycoformGeneratorBase(object):
    @classmethod
    def from_hypothesis(cls, session, hypothesis_id):
        glycan_combinations = GlycanCombinationRecord.from_hypothesis(session, hypothesis_id)
        return cls(glycan_combinations)

    def __init__(self, glycan_combinations, *args, **kwargs):
        if not isinstance(glycan_combinations, SearchableMassCollection):
            glycan_combinations = NeutralMassDatabase(
                list(glycan_combinations), operator.attrgetter("dehydrated_mass"))
        self.glycan_combinations = glycan_combinations
        super(GlycoformGeneratorBase, self).__init__(*args, **kwargs)

    def handle_glycan_combination(self, peptide_obj, peptide_record, glycan_combination,
                                  glycosylation_sites, core_type):
        glycosylation_sites_unoccupied = set(glycosylation_sites)
        for site in list(glycosylation_sites_unoccupied):
            if peptide_obj[site][1]:
                glycosylation_sites_unoccupied.remove(site)
        site_combinations = list(limiting_combinations(glycosylation_sites_unoccupied, glycan_combination.count))
        result_set = [None for i in site_combinations]
        key = self._make_key(peptide_record, glycan_combination)
        for i, site_set in enumerate(site_combinations):
            glycoform = peptide_obj.clone()
            glycoform.id = key
            glycoform.glycan = glycan_combination.composition.clone()
            for site in site_set:
                glycoform.add_modification(site, core_type.name)
            result_set[i] = glycoform
        return result_set

    def _make_key(self, peptide_record, glycan_combination):
        key = (peptide_record.start_position, peptide_record.end_position,
               peptide_record.protein_id, peptide_record.hypothesis_id,
               glycan_combination.id)
        return key

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
        super(PeptideGlycosylator, self).__init__(glycan_combinations, *args, **kwargs)
        if not isinstance(peptide_records, SearchableMassCollection):
            peptide_records = NeutralMassDatabase(peptide_records)
        self.peptides = peptide_records

    def handle_peptide_mass(self, peptide_mass, intact_mass, error_tolerance=1e-5):
        peptide_records = self.peptides.search_mass_ppm(peptide_mass, error_tolerance)
        glycan_mass = intact_mass - peptide_mass
        glycan_combinations = self.glycan_combinations.search_mass_ppm(glycan_mass, error_tolerance)
        result_set = []
        for peptide in peptide_records:
            self._combinate(peptide, glycan_combinations, result_set)
        return result_set

    def _combinate(self, peptide, glycan_combinations, result_set=None):
        if result_set is None:
            result_set = []
        peptide_obj = peptide.convert()
        for glycan_combination in glycan_combinations:
            for tp in glycan_combination.glycan_types:
                tp = GlycosylationType[tp]
                if tp is GlycosylationType.n_linked:
                    result_set.extend(
                        self.handle_n_glycan(peptide_obj.clone(), peptide, glycan_combination))
                elif tp is GlycosylationType.o_linked:
                    result_set.extend(
                        self.handle_o_glycan(peptide_obj.clone(), peptide, glycan_combination))
                elif tp is GlycosylationType.glycosaminoglycan:
                    result_set.extend(
                        self.handle_gag_linker(peptide_obj.clone(), peptide, glycan_combination))
        return result_set

    def generate_crossproduct(self, lower_bound=0, upper_bound=float('inf')):
        minimum_peptide_mass = max(lower_bound - self.glycan_combinations.highest_mass, 0)
        maximum_peptide_mass = max(upper_bound - self.glycan_combinations.lowest_mass, 0)
        peptides = self.peptides.search_between(minimum_peptide_mass - 100, maximum_peptide_mass)
        for peptide in peptides:
            if not peptide.has_glycosylation_sites():
                continue
            glycan_mass_limit = upper_bound - peptide.calculated_mass
            if glycan_mass_limit < 0:
                continue
            minimum_glycan_mass = max(lower_bound - peptide.calculated_mass, 0)
            glycan_combinations = self.glycan_combinations.search_between(minimum_glycan_mass, glycan_mass_limit + 10)
            for solution in self._combinate(peptide, glycan_combinations):
                total_mass = solution.total_mass
                if total_mass < lower_bound or total_mass > upper_bound:
                    continue
                yield solution
