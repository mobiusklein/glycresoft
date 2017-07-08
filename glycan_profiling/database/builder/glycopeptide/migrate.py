from uuid import uuid4

from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycopeptideHypothesis,
    Protein, Peptide, Glycopeptide, GlycanCombination,
    GlycanCombinationGlycanComposition)

from glycan_profiling.serialize.utils import get_uri_for_instance

from glycan_profiling.database.builder.glycan.migrate import GlycanHypothesisMigrator

from glycan_profiling.task import TaskBase


class GlycopeptideHypothesisMigrator(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.hypothesis_id = None
        self.glycan_hypothesis_id = None
        self._glycan_hypothesis_migrator = None
        self.protein_id_map = dict()
        self.peptide_id_map = dict()
        self.glycan_combination_id_map = dict()
        self.glycopeptide_id_map = dict()

    def _migrate(self, obj):
        self.session.add(obj)
        self.session.flush()
        new_id = obj.id
        return new_id

    def commit(self):
        self.session.commit()

    def migrate_hypothesis(self, hypothesis):
        new_inst = GlycopeptideHypothesis(
            name=hypothesis.name,
            uuid=str(uuid4().hex),
            status=hypothesis.status,
            glycan_hypothesis_id=None,
            parameters=dict(hypothesis.parameters))
        new_inst.parameters["original_uuid"] = hypothesis.uuid
        new_inst.parameters['copied_from'] = get_uri_for_instance(hypothesis)
        self.hypothesis_id = self._migrate(new_inst)
        self.create_glycan_hypothesis_migrator(hypothesis.glycan_hypothesis)
        new_inst.glycan_hypothesis_id = self.glycan_hypothesis_id
        self._migrate(new_inst)

    def migrate_glycan_composition(self, glycan_composition):
        self._glycan_hypothesis_migrator.migrate_glycan_composition(
            glycan_composition)

    def migrate_glycan_combination(self, glycan_combination):
        new_inst = GlycanCombination(
            calculated_mass=glycan_combination.calculated_mass,
            formula=glycan_combination.formula,
            composition=glycan_combination.composition,
            count=glycan_combination.count,
            hypothesis_id=self.hypothesis_id)
        self.glycan_combination_id_map[glycan_combination.id] = self._migrate(new_inst)
        component_acc = []
        for component, count in glycan_combination.components.add_column(
                GlycanCombinationGlycanComposition.c.count):
            new_component_id = self._glycan_hypothesis_migrator.glycan_composition_id_map[component.id]
            component_acc.append(
                {"glycan_id": new_component_id, "combination_id": new_inst.id, "count": count})
        if component_acc:
            self.session.execute(GlycanCombinationGlycanComposition.insert(), component_acc)

    def migrate_protein(self, protein):
        new_inst = Protein(
            name=protein.name,
            protein_sequence=protein.protein_sequence,
            other=dict(protein.other or dict()),
            hypothesis_id=self.hypothesis_id)
        self.protein_id_map[protein.id] = self._migrate(new_inst)

    def create_glycan_hypothesis_migrator(self, glycan_hypothesis):
        self._glycan_hypothesis_migrator = GlycanHypothesisMigrator(self._original_connection)
        self._glycan_hypothesis_migrator.bridge(self)
        self._glycan_hypothesis_migrator.migrate_hypothesis(glycan_hypothesis)
        self.glycan_hypothesis_id = self._glycan_hypothesis_migrator.hypothesis_id

    def migrate_peptide(self, peptide):
        new_inst = Peptide(
            calculated_mass=peptide.calculated_mass,
            base_peptide_sequence=peptide.base_peptide_sequence,
            modified_peptide_sequence=peptide.modified_peptide_sequence,
            formula=peptide.formula,
            count_glycosylation_sites=peptide.count_glycosylation_sites,
            count_missed_cleavages=peptide.count_missed_cleavages,
            count_variable_modifications=peptide.count_variable_modifications,
            start_position=peptide.start_position,
            end_position=peptide.end_position,
            peptide_score=peptide.peptide_score,
            peptide_score_type=peptide.peptide_score_type,
            sequence_length=peptide.sequence_length,
            # Update the protein id
            protein_id=self.protein_id_map[peptide.protein_id],
            # Update the hypothesis id
            hypothesis_id=self.hypothesis_id,
            n_glycosylation_sites=peptide.n_glycosylation_sites,
            o_glycosylation_sites=peptide.o_glycosylation_sites,
            gagylation_sites=peptide.gagylation_sites)
        self.peptide_id_map[peptide.id] = self._migrate(new_inst)

    def migrate_glycopeptide(self, glycopeptide):
        new_inst = Glycopeptide(
            calculated_mass=glycopeptide.calculated_mass,
            formula=glycopeptide.formula,
            glycopeptide_sequence=glycopeptide.glycopeptide_sequence,
            peptide_id=self.peptide_id_map[glycopeptide.peptide_id],
            protein_id=self.protein_id_map[glycopeptide.protein_id],
            hypothesis_id=self.hypothesis_id,
            glycan_combination_id=self.glycan_combination_id_map[glycopeptide.glycan_combination_id])
        self.glycopeptide_id_map[glycopeptide.id] = self._migrate(new_inst)
