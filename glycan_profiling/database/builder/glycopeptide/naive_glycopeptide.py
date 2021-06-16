
from glycan_profiling.serialize import func
from glycan_profiling.serialize.hypothesis.peptide import (
    Peptide, Protein, ProteinSite, Glycopeptide)
from glycan_profiling.serialize.utils import toggle_indices

from glycopeptidepy.algorithm import reverse_sequence
from glycopeptidepy.structure.sequence import (
    find_n_glycosylation_sequons, find_o_glycosylation_sequons,
    find_glycosaminoglycan_sequons, PeptideSequence)

from glycopeptidepy.structure.residue import UnknownAminoAcidException

from six import string_types as basestring

from .proteomics.peptide_permutation import (
    ProteinDigestor,
    MultipleProcessProteinDigestor,
    UniprotProteinAnnotator)
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
                 max_missed_cleavages=2, max_glycosylation_events=1, semispecific=False,
                 max_variable_modifications=None, full_cross_product=True,
                 peptide_length_range=(5, 60), require_glycosylation_sites=True):
        GlycopeptideHypothesisSerializerBase.__init__(
            self, connection, hypothesis_name, glycan_hypothesis_id, full_cross_product)
        self.fasta_file = fasta_file
        self.protease = protease
        self.constant_modifications = constant_modifications
        self.variable_modifications = variable_modifications
        self.max_missed_cleavages = max_missed_cleavages
        self.max_glycosylation_events = max_glycosylation_events
        self.semispecific = semispecific
        self.max_variable_modifications = max_variable_modifications
        self.peptide_length_range = peptide_length_range or (5, 60)
        self.require_glycosylation_sites = require_glycosylation_sites

        params = {
            "fasta_file": fasta_file,
            "enzymes": [protease] if isinstance(protease, basestring) else list(protease),
            "constant_modifications": constant_modifications,
            "variable_modifications": variable_modifications,
            "max_missed_cleavages": max_missed_cleavages,
            "max_glycosylation_events": max_glycosylation_events,
            "semispecific": semispecific,
            "max_variable_modifications": max_variable_modifications,
            "full_cross_product": self.full_cross_product,
            "peptide_length_range": self.peptide_length_range,
            "require_glycosylation_sites": self.require_glycosylation_sites,
        }
        self.set_parameters(params)

    def extract_proteins(self):
        i = 0
        for protein in ProteinFastaFileParser(self.fasta_file):
            protein.hypothesis_id = self.hypothesis_id
            protein._init_sites()

            self.session.add(protein)
            i += 1
            if i % 5000 == 0:
                self.log("... %d Proteins Extracted" % (i,))
                self.session.commit()

        self.session.commit()
        self.log("%d Proteins Extracted" % (i,))
        return i

    def protein_ids(self):
        return [i[0] for i in self.query(Protein.id).filter(Protein.hypothesis_id == self.hypothesis_id).all()]

    def digest_proteins(self):
        digestor = ProteinDigestor(
            self.protease, self.constant_modifications, self.variable_modifications,
            self.max_missed_cleavages, min_length=self.peptide_length_range[0],
            max_length=self.peptide_length_range[1], semispecific=self.semispecific,
            require_glycosylation_sites=self.require_glycosylation_sites)
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
                self.log("... %0.3f%% Complete (%d/%d). %d Peptides Produced." % (i * 100. / n, i, n, j))
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
        annotator = UniprotProteinAnnotator(
            self, self.protein_ids(), self.constant_modifications, self.variable_modifications)
        annotator.run()
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
                    self.session.bulk_insert_mappings(Glycopeptide, acc, render_nulls=True)
                    self.session.commit()
                    acc = []
        self.session.bulk_insert_mappings(Glycopeptide, acc, render_nulls=True)
        self.session.commit()

    def run(self):
        self.log("Extracting Proteins")
        self.extract_proteins()
        self.log("Digesting Proteins")
        index_toggler = toggle_indices(self.session, Peptide)
        index_toggler.drop()
        self.digest_proteins()
        self.log("Rebuilding Peptide Index")
        index_toggler.create()
        self.split_proteins()
        self.log("Combinating Glycans")
        self.combinate_glycans(self.max_glycosylation_events)
        if self.full_cross_product:
            self.log("Building Glycopeptides")
            self.glycosylate_peptides()
        self._sql_analyze_database()
        if self.full_cross_product:
            self._count_produced_glycopeptides()
        self.log("Done")


class MultipleProcessFastaGlycopeptideHypothesisSerializer(FastaGlycopeptideHypothesisSerializer):
    def __init__(self, fasta_file, connection, glycan_hypothesis_id, hypothesis_name=None,
                 protease='trypsin', constant_modifications=None, variable_modifications=None,
                 max_missed_cleavages=2, max_glycosylation_events=1, semispecific=False,
                 max_variable_modifications=None, full_cross_product=True, peptide_length_range=(5, 60),
                 require_glycosylation_sites=True,
                 n_processes=4):
        super(MultipleProcessFastaGlycopeptideHypothesisSerializer, self).__init__(
            fasta_file, connection, glycan_hypothesis_id, hypothesis_name,
            protease, constant_modifications, variable_modifications,
            max_missed_cleavages, max_glycosylation_events, semispecific,
            max_variable_modifications, full_cross_product, peptide_length_range,
            require_glycosylation_sites)
        self.n_processes = n_processes

    def digest_proteins(self):
        digestor = ProteinDigestor(
            self.protease, self.constant_modifications, self.variable_modifications,
            self.max_missed_cleavages, semispecific=self.semispecific,
            require_glycosylation_sites=self.require_glycosylation_sites)
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
            original_sequence = protein.protein_sequence
            n = len(original_sequence)
            if "(" in protein.protein_sequence:
                try:
                    protein.protein_sequence = str(reverse_sequence(protein.protein_sequence, suffix_len=0))
                except UnknownAminoAcidException:
                    continue
            else:
                protein.protein_sequence = protein.protein_sequence[::-1]
            protein.hypothesis_id = self.hypothesis_id
            sites = []
            try:
                original_sequence = PeptideSequence(original_sequence)
            except UnknownAminoAcidException:
                continue
            try:
                n_glycosites = find_n_glycosylation_sequons(original_sequence)
                for n_glycosite in n_glycosites:
                    sites.append(
                        ProteinSite(name=ProteinSite.N_GLYCOSYLATION, location=n - n_glycosite - 1))
            except UnknownAminoAcidException:
                pass

            # See Protein._init_sites for explanation
            # try:
            #     o_glycosites = find_o_glycosylation_sequons(original_sequence)
            #     for o_glycosite in o_glycosites:
            #         sites.append(
            #             ProteinSite(name=ProteinSite.O_GLYCOSYLATION, location=n - o_glycosite - 1))
            # except UnknownAminoAcidException:
            #     pass

            # try:
            #     gag_sites = find_glycosaminoglycan_sequons(original_sequence)
            #     for gag_site in gag_sites:
            #         sites.append(
            #             ProteinSite(name=ProteinSite.GAGYLATION, location=n - gag_site - 1))
            # except UnknownAminoAcidException:
            #     pass
            protein.sites.extend(sites)

            self.session.add(protein)
            i += 1
            if i % 5000 == 0:
                self.log("... %d Proteins Extracted" % (i,))
                self.session.commit()

        self.session.commit()
        self.log("%d Proteins Extracted" % (i,))
        return i


class NonSavingMultipleProcessFastaGlycopeptideHypothesisSerializer(_MPFGHS):
    def _spawn_glycosylator(self, input_queue, done_event):
        return NonSavingPeptideGlycosylatingProcess(
            self._original_connection, self.hypothesis_id, input_queue,
            chunk_size=3500, done_event=done_event)
