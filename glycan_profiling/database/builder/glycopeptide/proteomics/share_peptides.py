import logging
import re


from collections import defaultdict


from sqlalchemy.orm import make_transient
from glycan_profiling.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from .utils import slurp


from multiprocessing import Process, Queue, Event


logger = logging.getLogger("share_peptides")


class PeptideBlueprint(object):
    def __init__(self, base_peptide_sequence, modified_peptide_sequence, sequence_length,
                 calculated_mass, formula, count_glycosylation_sites, count_missed_cleavages,
                 count_variable_modifications, peptide_score, peptide_score_type, n_glycosylation_sites,
                 o_glycosylation_sites, gagylation_sites):
        self.base_peptide_sequence = base_peptide_sequence
        self.modified_peptide_sequence = modified_peptide_sequence
        self.sequence_length = sequence_length
        self.calculated_mass = calculated_mass
        self.formula = formula
        self.count_glycosylation_sites = count_glycosylation_sites
        self.count_missed_cleavages = count_missed_cleavages
        self.count_variable_modifications = count_variable_modifications
        self.peptide_score = peptide_score
        self.peptide_score_type = peptide_score_type
        self.n_glycosylation_sites = n_glycosylation_sites
        self.o_glycosylation_sites = o_glycosylation_sites
        self.gagylation_sites = gagylation_sites

    def as_db_instance(self):
        return Peptide(
            base_peptide_sequence=self.base_peptide_sequence, modified_peptide_sequence=self.modified_peptide_sequence,
            sequence_length=self.sequence_length,
            calculated_mass=self.calculated_mass, formula=self.formula,
            count_glycosylation_sites=self.count_glycosylation_sites,
            count_missed_cleavages=self.count_missed_cleavages,
            count_variable_modifications=self.count_variable_modifications, peptide_score=self.peptide_score,
            peptide_score_type=self.peptide_score_type, n_glycosylation_sites=self.n_glycosylation_sites,
            o_glycosylation_sites=self.o_glycosylation_sites,
            gagylation_sites=self.gagylation_sites)

    @classmethod
    def from_db_instance(cls, inst):
        return cls(
            base_peptide_sequence=inst.base_peptide_sequence, modified_peptide_sequence=inst.modified_peptide_sequence,
            sequence_length=inst.sequence_length,
            calculated_mass=inst.calculated_mass, formula=inst.formula,
            count_glycosylation_sites=inst.count_glycosylation_sites,
            count_missed_cleavages=inst.count_missed_cleavages,
            count_variable_modifications=inst.count_variable_modifications, peptide_score=inst.peptide_score,
            peptide_score_type=inst.peptide_score_type, n_glycosylation_sites=inst.n_glycosylation_sites,
            o_glycosylation_sites=inst.o_glycosylation_sites,
            gagylation_sites=inst.gagylation_sites)


class PeptideIndex(object):
    def __init__(self, store=None):
        if store is None:
            store = defaultdict(list)
        self.store = store

    def all_but(self, key):
        for bin_key, values in self.store.items():
            if key == bin_key:
                continue
            for v in values:
                yield v

    def populate(self, iterable):
        for peptide in iterable:
            self.store[peptide.protein_id].append(PeptideBlueprint.from_db_instance(peptide))


class PeptideSharer(DatabaseBoundOperation):
    def __init__(self, connection, hypothesis_id):
        DatabaseBoundOperation.__init__(self, connection)
        self.hypothesis_id = hypothesis_id
        self.index = PeptideIndex()
        self.index.populate(self._get_all_peptides())

    def find_contained_peptides(self, target_protein):
        session = self.session

        protein_id = target_protein.id

        i = 0
        j = 0
        # logger.info("Enriching %r", target_protein)
        target_protein_sequence = target_protein.protein_sequence
        keepers = []
        for peptide in self.stream_distinct_peptides(target_protein):

            match = self.decider_fn(peptide.base_peptide_sequence, target_protein_sequence)

            if match is not False:
                start, end, distance = match
                peptide = peptide.as_db_instance()
                peptide.hypothesis_id = self.hypothesis_id
                peptide.protein_id = protein_id
                peptide.start_position = start
                peptide.end_position = end
                keepers.append(peptide)
                j += 1
            i += 1
            if i % 100000 == 0:
                logger.info("%d peptides handled for %r", i, target_protein)
            if len(keepers) > 1000:
                session.bulk_save_objects(keepers)
                session.commit()
                keepers = []
                logger.info("%d peptides shared for %r", j, target_protein)

        session.bulk_save_objects(keepers)
        session.commit()
        # session.expunge_all()

    def decider_fn(self, query_seq, target_seq, **kwargs):
        match = re.search(query_seq, target_seq)
        if match:
            return match.start(), match.end(), 0
        return False

    def _get_all_peptides(self):
        return self.session.query(Peptide).filter(
            Peptide.hypothesis_id == self.hypothesis_id).all()

    def stream_distinct_peptides(self, protein):
        q = self.index.all_but(protein.id)

        for i in q:
            yield i


class PeptideSharingProcess(Process):
    def __init__(self, connection, hypothesis_id, input_queue, done_event=None):
        Process.__init__(self)
        self.connection = connection
        self.input_queue = input_queue
        self.hypothesis_id = hypothesis_id
        self.done_event = done_event

    def task(self):
        database = DatabaseBoundOperation(self.connection)
        session = database.session
        has_work = True

        sharer = PeptideSharer(self.connection, self.hypothesis_id)

        while has_work:
            try:
                work_items = self.input_queue.get(timeout=5)
                if work_items is None:
                    has_work = False
                    continue
            except:
                if self.done_event.is_set():
                    has_work = False
                continue
            proteins = slurp(session, Protein, work_items, flatten=False)
            for protein in proteins:
                sharer.find_contained_peptides(protein)

    def run(self):
        self.task()


class MultipleProcessPeptideSharer(object):
    def __init__(self, connection, hypothesis_id, protein_ids, n_processes=4):
        self.connection = connection
        self.hypothesis_id = hypothesis_id
        self.protein_ids = protein_ids
        self.n_processes = n_processes

    def run(self):
        input_queue = Queue(100)
        done_event = Event()
        processes = [
            PeptideSharingProcess(
                self.connection, self.hypothesis_id, input_queue,
                done_event=done_event) for i in range(self.n_processes)
        ]
        protein_ids = self.protein_ids
        i = 0
        chunk_size = 20
        for process in processes:
            input_queue.put(protein_ids[i:(i + chunk_size)])
            i += chunk_size
            process.start()

        while i < len(protein_ids):
            input_queue.put(protein_ids[i:(i + chunk_size)])
            i += chunk_size

        done_event.set()
        for process in processes:
            process.join()
