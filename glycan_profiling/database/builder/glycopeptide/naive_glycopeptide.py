from multiprocessing import Queue, Event
from glycan_profiling.serialize.hypothesis.peptide import Peptide, Protein

from .proteomics.peptide_permutation import ProteinDigestor
from .proteomics.fasta import ProteinFastaFileParser
from .common import (
    GlycopeptideHypothesisSerializerBase, DatabaseBoundOperation,
    PeptideGlycosylator, PeptideGlycosylatingProcess)


class FastaGlycopeptideHypothesisSerializer(GlycopeptideHypothesisSerializerBase):
    def __init__(self, fasta_file, connection, glycan_hypothesis_id, hypothesis_name=None,
                 protease='trypsin', constant_modifications=None, variable_modifications=None,
                 max_missed_cleavages=2, max_glycosylation_events=1):
        GlycopeptideHypothesisSerializerBase.__init__(self, connection, hypothesis_name, glycan_hypothesis_id)
        self.fasta_file = fasta_file
        self.protease = protease
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.max_missed_cleavages = max_missed_cleavages
        self.max_glycosylation_events = max_glycosylation_events

    def extract_proteins(self):
        for protein in ProteinFastaFileParser(self.fasta_file):
            protein.hypothesis_id = self.hypothesis_id
            self.session.add(protein)
        self.session.commit()

    def protein_ids(self):
        return [i[0] for i in self.query(Protein.id).filter(Protein.hypothesis_id == self.hypothesis_id).all()]

    def peptide_ids(self):
        return [i[0] for i in self.query(Peptide.id).filter(Peptide.hypothesis_id == self.hypothesis_id).all()]

    def digest_proteins(self):
        digestor = ProteinDigestor(
            self.protease, self.constant_modifications, self.variable_modifications,
            self.max_missed_cleavages)
        for protein_id in self.protein_ids():
            protein = self.query(Protein).get(protein_id)
            acc = []
            for peptide in digestor.process_protein(protein):
                acc.append(peptide)
                if len(acc) > 100000:
                    self.session.add_all(acc)
                    self.session.commit()
                    acc = []
            self.session.add_all(acc)
            self.session.commit()
            acc = []

    def glycosylate_peptides(self):
        glycosylator = PeptideGlycosylator(self.session, self.hypothesis_id)
        acc = []
        i = 0
        for peptide_id in self.peptide_ids():
            peptide = self.query(Peptide).get(peptide_id)
            for glycopeptide in glycosylator.handle_peptide(peptide):
                acc.append(glycopeptide)
                i += 1
                if len(acc) > 100000:
                    self.session.add_all(acc)
                    self.session.commit()
                    acc = []
        self.session.add_all(acc)
        self.session.commit()

    def run(self):
        self.log("Extracting Proteins")
        self.extract_proteins()
        self.log("Digesting Proteins")
        self.digest_proteins()
        self.log("Combinating Glycans")
        self.combinate_glycans(self.max_glycosylation_events)
        self.log("Building Glycopeptides")
        self.glycosylate_peptides()
        self._count_produced_glycopeptides()
        self.log("Done")


class MultipleProcessFastaGlycopeptideHypothesisSerializer(FastaGlycopeptideHypothesisSerializer):
    def __init__(self, fasta_file, connection, glycan_hypothesis_id, hypothesis_name=None,
                 protease='trypsin', constant_modifications=None, variable_modifications=None,
                 max_missed_cleavages=2, max_glycosylation_events=1, n_processes=4):
        super(MultipleProcessFastaGlycopeptideHypothesisSerializer, self).__init__(
            fasta_file, connection, glycan_hypothesis_id, hypothesis_name,
            protease, constant_modifications, variable_modifications,
            max_missed_cleavages, max_glycosylation_events)
        self.n_processes = n_processes

    def glycosylate_peptides(self):
        input_queue = Queue(100)
        done_event = Event()
        processes = [
            PeptideGlycosylatingProcess(
                self._original_connection, self.hypothesis_id, input_queue,
                chunk_size=15000, done_event=done_event) for i in range(self.n_processes)
        ]
        peptide_ids = self.peptide_ids()
        i = 0
        chunk_size = 20
        for process in processes:
            input_queue.put(peptide_ids[i:(i + chunk_size)])
            i += chunk_size
            process.start()

        while i < len(peptide_ids):
            input_queue.put(peptide_ids[i:(i + chunk_size)])
            i += chunk_size

        done_event.set()
        for process in processes:
            process.join()
