'''Auxiliary data structures used to look up records from a database
using the Sequence interface or the Mapping interface for convenience.

Maintainence Note: Mostly unused outside of debugging
'''

from glycresoft.serialize import (
    Protein, Peptide, Glycopeptide)

from glycresoft.structure import LRUCache


class ProteinIndex(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id

    def _get_by_id(self, id):
        return self.session.query(Protein).get(id)

    def _get_by_name(self, name):
        return self.session.query(Protein).filter(
            Protein.hypothesis_id == self.hypothesis_id,
            Protein.name == name).one()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_id(key)
        else:
            return self._get_by_name(key)

    def __iter__(self):
        q = self.session.query(Protein).filter(Protein.hypothesis_id == self.hypothesis_id)
        return iter(q)

    def __len__(self):
        return self.session.query(Protein).filter(Protein.hypothesis_id == self.hypothesis_id).count()


class PeptideIndex(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.hypothesis_id = hypothesis_id

    def _get_by_id(self, id):
        return self.session.query(Peptide).get(id)

    def _get_by_sequence(self, modified_peptide_sequence, protein_id):
        return self.session.query(Peptide).filter(
            Peptide.hypothesis_id == self.hypothesis_id,
            Peptide.modified_peptide_sequence == modified_peptide_sequence,
            Peptide.protein_id == protein_id).one()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_id(key)
        else:
            return self._get_by_sequence(*key)

    def __iter__(self):
        q = self.session.query(Peptide).filter(Peptide.hypothesis_id == self.hypothesis_id)
        return iter(q)

    def __len__(self):
        return self.session.query(Peptide).filter(Peptide.hypothesis_id == self.hypothesis_id).count()
