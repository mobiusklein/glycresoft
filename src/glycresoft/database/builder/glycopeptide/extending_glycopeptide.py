from glypy.composition import formula

from .proteomics.peptide_permutation import peptide_permutations
from glycresoft.serialize import Peptide, DatabaseBoundOperation


class ProteomeCollection(object):
    def __init__(self, proteins, peptides):
        self.proteins = proteins
        self.peptides = peptides

    def add_modifications(self, constant_modifications=None, variable_modifications=None, max_variable_modifications=4):
        if constant_modifications is None:
            constant_modifications = []
        if variable_modifications is None:
            variable_modifications = []
        result = MemoryPeptideCollection()
        for peptide in self.peptides:
            for modified_peptide, n_variable_modifications in peptide_permutations(
                    str(peptide), constant_modifications, variable_modifications):
                total_modification_count = (
                    n_variable_modifications + peptide.count_variable_modifications)
                if total_modification_count > max_variable_modifications:
                    continue
                inst = Peptide(
                    base_peptide_sequence=peptide.base_peptide_sequence,
                    modified_peptide_sequence=str(modified_peptide),
                    count_missed_cleavages=peptide.count_missed_cleavages,
                    count_variable_modifications=total_modification_count,
                    sequence_length=peptide.sequence_length,
                    start_position=peptide.start_position,
                    end_position=peptide.end_position,
                    calculated_mass=modified_peptide.mass,
                    formula=formula(modified_peptide.total_composition()))
                result.add(inst)
        return result


class PeptideCollectionBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        raise NotImplementedError()

    def add(self, peptide):
        raise NotImplementedError()

    def update(self, peptides):
        for peptide in peptides:
            self.add(peptide)

    def save(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __repr__(self):
        return "{self.__class__.__name__}({size})".format(self=self, size=len(self))


class MemoryPeptideCollection(PeptideCollectionBase):
    def __init__(self, storage=None, *args, **kwargs):
        PeptideCollectionBase.__init__(self, *args, **kwargs)
        self.storage = list(storage or [])

    def __iter__(self):
        return iter(self.storage)

    def __len__(self):
        return len(self.storage)

    def add(self, peptide):
        self.storage.append(peptide)

    def save(self):
        pass


class DatabasePeptideCollection(PeptideCollectionBase, DatabaseBoundOperation):
    def __init__(self, connection, hypothesis_id=None, *args, **kwargs):
        if hypothesis_id is None:
            hypothesis_id = 1
        DatabaseBoundOperation.__init__(self, connection)
        PeptideCollectionBase.__init__(self, *args, **kwargs)
        self.hypothesis_id = hypothesis_id
        self._operation_count = 0
        self._batch_size = int(kwargs.get("batch_size", 1000))

    def __iter__(self):
        q = self._get_query()
        q = q.yield_per(5000)
        return q

    def _get_query(self):
        q = self.session.query(Peptide).filter(Peptide.hypothesis_id == self.hypothesis_id)
        return q

    def __len__(self):
        return self._get_query().count()

    def _add(self, peptide):
        peptide.hypothesis_id = self.hypothesis_id
        self.session.add(peptide)
        self._operation_count += 1

    def add(self, peptide):
        self._add(peptide)
        self._flush_if_work_done()

    def update(self, peptides):
        for peptide in peptides:
            self._add(peptide)
        self._flush_if_work_done()

    def _flush_if_work_done(self):
        if self._operation_count > self._batch_size:
            self.session.flush()

    def save(self):
        self.session.commit()
