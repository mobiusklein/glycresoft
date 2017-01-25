import re
import textwrap

from glycopeptidepy.io.fasta import FastaFileParser, FastaFileWriter

from glycan_profiling.serialize.hypothesis.peptide import Protein
from .sequence_tree import SuffixTree


def tryopen(path):
    if hasattr(path, 'read'):
        return path
    return open(path)


class ProteinFastaFileParser(FastaFileParser):

    def __init__(self, path):
        super(ProteinFastaFileParser, self).__init__(path)

    def process_result(self, d):
        p = Protein(name=str(d['name']), protein_sequence=d['sequence'])
        return p


class ProteinFastaFileWriter(FastaFileWriter):

    def write(self, protein):
        defline = ''.join(
            [">", protein.name, " ", str(protein.glycosylation_sites)])
        seq = '\n'.join(textwrap.wrap(protein.protein_sequence, 80))
        super(ProteinFastaFileWriter, self).write(defline, seq)

    def writelines(self, iterable):
        for protein in iterable:
            self.write(protein)


class SequenceLabel(object):

    def __init__(self, label, sequence):
        self.label = label
        self.sequence = sequence

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.label[i]
        else:
            return SequenceLabel(self.label[i], self.sequence)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return iter(self.label)

    def __repr__(self):
        return "SequenceLabel(%s, %r)" % (self.label, self.sequence)


class ProteinSequenceListResolver(object):

    @classmethod
    def build_from_fasta(cls, fasta):
        return cls(ProteinFastaFileParser(fasta))

    def __init__(self, protein_list):
        protein_list = [SequenceLabel(str(prot.name), prot)
                        for prot in protein_list]
        self.resolver = SuffixTree()
        for prot in protein_list:
            self.resolver.add_ngram(prot)

    def find(self, name):
        return [label_seq.sequence for label_seq in self.resolver.subsequences_of(name)]

    def __getitem__(self, name):
        return self.find(name)
