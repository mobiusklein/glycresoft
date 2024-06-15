from glycresoft.serialize.utils import temp_table
from glycresoft.serialize import (
    Peptide, Protein, DatabaseBoundOperation,
    TemplateNumberStore)

from glycresoft.task import TaskBase


class DeduplicatePeptides(DatabaseBoundOperation, TaskBase):
    def __init__(self, connection, hypothesis_id):
        DatabaseBoundOperation.__init__(self, connection)
        self.hypothesis_id = hypothesis_id

    def run(self):
        self.remove_duplicates()

    def find_best_peptides(self):
        q = self.session.query(
            Peptide.id, Peptide.peptide_score,
            Peptide.modified_peptide_sequence, Peptide.protein_id, Peptide.start_position).join(
            Protein).filter(Protein.hypothesis_id == self.hypothesis_id).yield_per(10000)
        keepers = dict()
        for id, score, modified_peptide_sequence, protein_id, start_position in q:
            key = (modified_peptide_sequence, protein_id, start_position)
            result = keepers.get(key)
            if result is None:
                keepers[key] = (id, score)
            elif result[1] < score:
                keepers[key] = (id, score)
        return keepers

    def store_best_peptides(self, keepers):
        table = temp_table(TemplateNumberStore)
        conn = self.session.connection()
        table.create(conn)
        payload = [{"value": x[0]} for x in keepers.values()]
        conn.execute(table.insert(), payload)
        self.session.commit()
        return table

    def remove_duplicates(self):
        self.log("... Deduplicating Peptides")
        keepers = self.find_best_peptides()
        self.log("... Marking Candidates")
        table = self.store_best_peptides(keepers)
        ids = self.session.query(table.c.value)
        self.log("... Building Mask")
        q = self.session.query(Peptide.id).filter(
            Peptide.protein_id == Protein.id,
            Protein.hypothesis_id == self.hypothesis_id,
            ~Peptide.id.in_(ids.correlate(None)))
        self.log("... Removing Duplicates")
        self.session.execute(Peptide.__table__.delete(
            Peptide.__table__.c.id.in_(q.selectable)))
        conn = self.session.connection()
        table.drop(conn)
        self.log("... Complete")
        self.session.commit()
