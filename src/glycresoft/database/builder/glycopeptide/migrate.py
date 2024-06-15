from uuid import uuid4

from glycresoft.serialize import (
    DatabaseBoundOperation, GlycopeptideHypothesis,
    Protein, Peptide, Glycopeptide, GlycanCombination,
    GlycanCombinationGlycanComposition, ProteinSite)

from glycresoft.serialize.utils import get_uri_for_instance

from glycresoft.database.builder.glycan.migrate import GlycanHypothesisMigrator

from glycresoft.task import TaskBase


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

    def clear(self):
        self.protein_id_map.clear()
        self.peptide_id_map.clear()
        self.glycan_combination_id_map.clear()
        self.glycopeptide_id_map.clear()

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
        for component, count in glycan_combination.components.add_columns(
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
        new_id = self._migrate(new_inst)
        self.protein_id_map[protein.id] = new_id
        for position in protein.sites.all():
            self.session.add(
                ProteinSite(location=position.location, name=position.name, protein_id=new_id))
        self.session.flush()

    def create_glycan_hypothesis_migrator(self, glycan_hypothesis):
        self._glycan_hypothesis_migrator = GlycanHypothesisMigrator(self._original_connection)
        self._glycan_hypothesis_migrator.bridge(self)
        self._glycan_hypothesis_migrator.migrate_hypothesis(glycan_hypothesis)
        self.glycan_hypothesis_id = self._glycan_hypothesis_migrator.hypothesis_id

    def migrate_peptides_bulk(self, peptides):
        reverse_map = {}
        n = len(peptides)
        buffer = []
        for i, peptide in enumerate(peptides):
            if i % 15000 == 0 and i:
                self.log("... Migrating Peptides %d/%d (%0.2f%%)" %
                         (i, n, i * 100.0 / n))
                self.session.bulk_save_objects(buffer)
                buffer = []
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
            buffer.append(new_inst)
            reverse_map[(peptide.modified_peptide_sequence,
                         self.protein_id_map[peptide.protein_id],
                         peptide.start_position, )] = peptide.id

        if buffer:
            self.session.bulk_save_objects(buffer)
            buffer = []

        self.session.flush()
        self.log("... Reconstructing Reverse Mapping")
        migrated = self.session.query(
            Peptide.id, Peptide.modified_peptide_sequence,
            Peptide.protein_id, Peptide.start_position).filter(
                Peptide.hypothesis_id == self.hypothesis_id).yield_per(10000)
        m = 0
        for peptide in migrated:
            m += 1
            self.peptide_id_map[
                reverse_map[peptide.modified_peptide_sequence,
                            peptide.protein_id, peptide.start_position]] = peptide.id
        if m != n:
            raise ValueError("Peptide Migration Unsuccessful")
        return m

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

    def migrate_glycopeptide_bulk(self, glycopeptides):
        reverse_map = {}
        n = len(glycopeptides)
        buffer = []
        for i, glycopeptide in enumerate(glycopeptides):
            if i % 50000 == 0 and i:
                self.log("... Migrating Glycopeptide %d/%d (%0.2f%%)" % (i, n, i * 100.0 / n))
                self.session.bulk_save_objects(buffer)
                buffer = []
            new_inst = Glycopeptide(
                calculated_mass=glycopeptide.calculated_mass,
                formula=glycopeptide.formula,
                glycopeptide_sequence=glycopeptide.glycopeptide_sequence,
                peptide_id=self.peptide_id_map[glycopeptide.peptide_id],
                protein_id=self.protein_id_map[glycopeptide.protein_id],
                hypothesis_id=self.hypothesis_id,
                glycan_combination_id=self.glycan_combination_id_map[glycopeptide.glycan_combination_id])
            buffer.append(new_inst)
            reverse_map[(glycopeptide.glycopeptide_sequence,
                         self.peptide_id_map[glycopeptide.peptide_id],
                         self.glycan_combination_id_map[glycopeptide.glycan_combination_id], )] = glycopeptide.id

        if buffer:
            self.session.bulk_save_objects(buffer)
            buffer = []

        self.session.flush()
        self.log("... Reconstructing Reverse Mapping")
        migrated = self.session.query(
            Glycopeptide.id, Glycopeptide.glycopeptide_sequence,
            Glycopeptide.peptide_id, Glycopeptide.glycan_combination_id).filter(
                Glycopeptide.hypothesis_id == self.hypothesis_id).yield_per(10000)

        m = 0
        for glycopeptide in migrated:
            m += 1
            self.glycopeptide_id_map[
                reverse_map[glycopeptide.glycopeptide_sequence,
                            glycopeptide.peptide_id,
                            glycopeptide.glycan_combination_id]] = glycopeptide.id
        if m != n:
            raise ValueError("Glycopeptide Migration Unsuccessful")
        return m

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
