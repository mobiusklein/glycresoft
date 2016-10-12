import logging
import re

from sqlalchemy.orm import make_transient
from glycan_profiling.serialize import (
    Peptide, Protein, DatabaseBoundOperation)

from .utils import slurp


from multiprocessing import Process, Queue, Event


logger = logging.getLogger("share_peptides")


class PeptideSharer(DatabaseBoundOperation):
    def __init__(self, connection, hypothesis_id):
        DatabaseBoundOperation.__init__(self, connection)
        self.hypothesis_id = hypothesis_id

    def find_contained_peptides(self, target_protein):
        session = self.session

        protein_id = target_protein.id

        i = 0
        logger.info("Enriching %r", target_protein)
        target_protein_sequence = target_protein.protein_sequence
        keepers = []
        for peptide in self.stream_distinct_peptides(target_protein):

            match = self.decider_fn(peptide.base_peptide_sequence, target_protein_sequence)

            if match is not False:
                start, end, distance = match
                make_transient(peptide)
                peptide.id = None
                peptide.protein_id = protein_id
                peptide.start_position = start
                peptide.end_position = end
                keepers.append(peptide)
            i += 1
            if i % 1000 == 0:
                logger.info("%d peptides handled for %r", i, target_protein)
            if len(keepers) > 1000:
                session.add_all(keepers)
                session.commit()
                keepers = []

        session.add_all(keepers)
        session.commit()

    def decider_fn(self, query_seq, target_seq, **kwargs):
        match = re.search(query_seq, target_seq)
        if match:
            return match.start(), match.end(), 0
        return False

    def stream_distinct_peptides(self, protein):
        q = self.session.query(Peptide).join(Protein).filter(
            Protein.hypothesis_id == self.hypothesis_id).distinct(
            Peptide.modified_peptide_sequence)

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
