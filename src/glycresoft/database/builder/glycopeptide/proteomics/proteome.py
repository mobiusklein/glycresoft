from glycresoft.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from .remove_duplicate_peptides import DeduplicatePeptides


class ProteomeBase(DatabaseBoundOperation):
    def __init__(self, connection, hypothesis_id, target_proteins=None,
                 constant_modifications=None, variable_modifications=None):
        if constant_modifications is None:
            constant_modifications = []
        if variable_modifications is None:
            variable_modifications = []
        DatabaseBoundOperation.__init__(self, connection)
        self.hypothesis_id = hypothesis_id
        self.target_proteins = target_proteins
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications

    def retrieve_target_protein_ids(self):
        if len(self.target_proteins) == 0:
            return [
                i[0] for i in
                self.query(Protein.id).filter(
                    Protein.hypothesis_id == self.hypothesis_id).all()
            ]
        else:
            result = []
            for target in self.target_proteins:
                if isinstance(target, str):
                    match = self.query(Protein.id).filter(
                        Protein.name == target,
                        Protein.hypothesis_id == self.hypothesis_id).first()
                    if match:
                        result.append(match[0])
                    else:
                        self.log("Could not locate protein '%s'" % target)
                elif isinstance(target, int):
                    result.append(target)
            return result

    def get_target_proteins(self):
        ids = self.retrieve_target_protein_ids()
        return [self.session.query(Protein).get(i) for i in ids]

    def count_peptides(self):
        peptide_count = self.session.query(Peptide).filter(
            Peptide.hypothesis_id == self.hypothesis_id).count()
        return peptide_count

    def remove_duplicates(self):
        DeduplicatePeptides(self._original_connection, self.hypothesis_id).run()
