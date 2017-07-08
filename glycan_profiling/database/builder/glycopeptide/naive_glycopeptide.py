
from glycan_profiling.serialize import func
from glycan_profiling.serialize.hypothesis.peptide import Peptide, Protein

from glycopeptidepy.algorithm import reverse_preserve_sequon

from six import string_types as basestring

from .proteomics.peptide_permutation import (
    ProteinDigestor,
    MultipleProcessProteinDigestor,
    ProteinSplitter)
from .proteomics.remove_duplicate_peptides import DeduplicatePeptides

from .proteomics.fasta import ProteinFastaFileParser
from .common import (
    GlycopeptideHypothesisSerializerBase,
    PeptideGlycosylator, PeptideGlycosylatingProcess,
    NonSavingPeptideGlycosylatingProcess,
    MultipleProcessPeptideGlycosylator)


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

        params = {
            "fasta_file": fasta_file,
            "enzymes": [protease] if isinstance(protease, basestring) else list(protease),
            "constant_modifications": constant_modifications,
            "variable_modifications": variable_modifications,
            "max_missed_cleavages": max_missed_cleavages,
            "max_glycosylation_events": max_glycosylation_events
        }
        self.set_parameters(params)

    def extract_proteins(self):
        i = 0
        for protein in ProteinFastaFileParser(self.fasta_file):
            protein.hypothesis_id = self.hypothesis_id
            self.session.add(protein)
            i += 1
            if i % 10000 == 0:
                self.log("%d Proteins Extracted" % (i,))
                self.session.commit()

        self.session.commit()

    def protein_ids(self):
        return [i[0] for i in self.query(Protein.id).filter(Protein.hypothesis_id == self.hypothesis_id).all()]

    def peptide_ids(self):
        return [i[0] for i in self.query(Peptide.id).filter(Peptide.hypothesis_id == self.hypothesis_id).all()]

    def digest_proteins(self):
        digestor = ProteinDigestor(
            self.protease, self.constant_modifications, self.variable_modifications,
            self.max_missed_cleavages)
        i = 0
        j = 0
        protein_ids = self.protein_ids()
        n = len(protein_ids)
        interval = min(n / 10., 100000)
        acc = []
        for protein_id in protein_ids:
            i += 1
            protein = self.query(Protein).get(protein_id)
            if i % interval == 0:
                self.log("%0.3f%% Complete (%d/%d). %d Peptides Produced." % (i * 100. / n, i, n, j))
            for peptide in digestor.process_protein(protein):
                acc.append(peptide)
                j += 1
                if len(acc) > 100000:
                    self.session.bulk_save_objects(acc)
                    self.session.commit()
                    acc = []
        self.session.bulk_save_objects(acc)
        self.session.commit()
        acc = []

    def split_proteins(self):
        self.log("Begin Applying Protein Annotations")
        splitter = ProteinSplitter(
            self.constant_modifications, self.variable_modifications)
        i = 0
        j = 0
        protein_ids = self.protein_ids()
        n = len(protein_ids)
        interval = min(n / 10., 100000)
        acc = []
        for protein_id in protein_ids:
            i += 1
            protein = self.query(Protein).get(protein_id)
            if i % interval == 0:
                self.log("%0.3f%% Complete (%d/%d). %d Peptides Produced." % (i * 100. / n, i, n, j))
            for peptide in splitter.handle_protein(protein):
                acc.append(peptide)
                j += 1
                if len(acc) > 100000:
                    self.session.bulk_save_objects(acc)
                    self.session.commit()
                    acc = []
        self.session.bulk_save_objects(acc)
        self.session.commit()
        acc = []
        DeduplicatePeptides(self._original_connection, self.hypothesis_id).run()

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
                    self.session.bulk_save_objects(acc)
                    self.session.commit()
                    acc = []
        self.session.bulk_save_objects(acc)
        self.session.commit()

    def run(self):
        self.log("Extracting Proteins")
        self.extract_proteins()
        self.log("Digesting Proteins")
        self.digest_proteins()
        self.split_proteins()
        self.log("Combinating Glycans")
        self.combinate_glycans(self.max_glycosylation_events)
        self.log("Building Glycopeptides")
        self.glycosylate_peptides()
        self._sql_analyze_database()
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

    def digest_proteins(self):
        digestor = ProteinDigestor(
            self.protease, self.constant_modifications, self.variable_modifications,
            self.max_missed_cleavages)
        task = MultipleProcessProteinDigestor(
            self._original_connection,
            self.hypothesis_id,
            self.protein_ids(),
            digestor, n_processes=self.n_processes)
        task.run()
        n_peptides = self.query(func.count(Peptide.id)).filter(
            Peptide.hypothesis_id == self.hypothesis_id).scalar()
        self.log("%d Base Peptides Produced" % (n_peptides,))

    def _spawn_glycosylator(self, input_queue, done_event):
        return PeptideGlycosylatingProcess(
            self._original_connection, self.hypothesis_id, input_queue,
            chunk_size=3500, done_event=done_event)

    def glycosylate_peptides(self):
        dispatcher = MultipleProcessPeptideGlycosylator(
            self._original_connection, self.hypothesis_id,
            glycan_combination_count=self.total_glycan_combination_count,
            n_processes=self.n_processes)
        dispatcher.process(self.peptide_ids())


_MPFGHS = MultipleProcessFastaGlycopeptideHypothesisSerializer


class ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer(_MPFGHS):
    def extract_proteins(self):
            i = 0
            for protein in ProteinFastaFileParser(self.fasta_file):
                protein.protein_sequence = str(reverse_preserve_sequon(protein.protein_sequence))
                protein.hypothesis_id = self.hypothesis_id
                self.session.add(protein)
                i += 1
                if i % 10000 == 0:
                    self.log("%d Proteins Extracted" % (i,))
                    self.session.commit()

            self.session.commit()


class NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer(_MPFGHS):
    def _spawn_glycosylator(self, input_queue, done_event):
        return NonSavingPeptideGlycosylatingProcess(
            self._original_connection, self.hypothesis_id, input_queue,
            chunk_size=3500, done_event=done_event)
