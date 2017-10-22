import re
import textwrap
from io import BytesIO, TextIOWrapper

from collections import OrderedDict

from glycopeptidepy.io.fasta import FastaFileParser, FastaFileWriter

from glycan_profiling.serialize.hypothesis.peptide import Protein
from .sequence_tree import SuffixTree


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


class FastaIndex(object):
    def __init__(self, path, block_size=1000000, encoding='utf-8'):
        self.path = path
        self.block_size = block_size
        self.index = OrderedDict()
        self.encoding = encoding

        self.build_index()

    def _chunk_iterator(self):
        delim = b"\n>"
        read_size = self.block_size
        with open(self.path, 'rb') as f:
            buff = f.read(read_size)
            started_with_with_delim = buff.startswith(delim)
            parts = buff.split(delim)
            tail = parts[-1]
            front = parts[:-1]
            i = 0
            for part in front:
                i += 1
                if part == b'':
                    continue
                if i == 1:
                    if started_with_with_delim:
                        yield delim + part
                    else:
                        yield part
                else:
                    yield delim + part
            running = True
            while running:
                buff = f.read(read_size)
                if len(buff) == 0:
                    running = False
                    buff = tail
                else:
                    buff = tail + buff
                parts = buff.split(delim)
                tail = parts[-1]
                front = parts[:-1]
                for part in front:
                    yield delim + part
            yield delim + tail

    def _generate_offsets(self):
        i = 0
        defline_pattern = re.compile(br"\s*>([^\n]+)\n")
        for line in self._chunk_iterator():
            match = defline_pattern.match(line)
            if match:
                yield i, match.group(1)
            i += len(line)
        yield i, None

    def build_index(self):
        index = OrderedDict()
        g = self._generate_offsets()
        last_offset = 0
        last_defline = None
        for offset, defline in g:
            if last_defline is not None:
                index[last_defline] = (last_offset, offset)
            last_defline = defline
            last_offset = offset
        assert last_defline is None
        self.index = index

    def _ensure_bytes(self, string):
        if isinstance(string, bytes):
            return string
        try:
            return string.encode(self.encoding)
        except (AttributeError, UnicodeEncodeError):
            raise TypeError("{!r} could not be encoded".format(string))

    def _offset_of(self, key):
        key = self._ensure_bytes(key)
        return self.index[key]

    def __len__(self):
        return len(self.index)

    def keys(self):
        return self.index.keys()

    def __iter__(self):
        return iter(self.index)

    def fetch_sequence(self, key):
        start, end = self._offset_of(key)
        with open(self.path, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)
            buff = TextIOWrapper(BytesIO(data))
            return next(ProteinFastaFileParser(buff))

    def __getitem__(self, key):
        return self.fetch_sequence(key)


class DeflineSuffix(object):
    def __init__(self, label, original):
        self.label = label
        self.original = original

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.label[i]
        else:
            return DeflineSuffix(self.label[i], self.original)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return iter(self.label)

    def __repr__(self):
        return "DeflineSuffix(%s, %r)" % (self.label, self.original)


class FastaProteinSequenceResolver(object):
    def __init__(self, path, encoding='utf-8', strategy='exact'):
        self.path = path
        self.encoding = encoding
        self.index = FastaIndex(self.path, encoding=self.encoding)
        self.resolver = None
        self.strategy = strategy

        self._build_resolver(strategy)

    def _build_resolver(self, strategy='exact'):
        if strategy == 'suffix':
            keys = self.index.keys()
            resolver = SuffixTree()
            i = 0
            for key in keys:
                i += 1
                label = DeflineSuffix(key.decode(self.encoding).split(" ")[0], key)
                resolver.add_ngram(label)
            self.resolver = resolver
        elif strategy == 'exact':
            keys = self.index.keys()
            resolver = dict()
            i = 0
            for key in keys:
                i += 1
                label = DeflineSuffix(key.decode(self.encoding).split(" ")[0], key)
                resolver[label.label] = label
            self.resolver = resolver

    def find(self, name):
        if self.strategy == 'suffix':
            labels = self.resolver.subsequences_of(name)
            sequences = [
                self.index[label.original] for label in labels]
        else:
            try:
                label = self.resolver[name]
                return [self.index[label.original]]
            except KeyError:
                return []
        return sequences

    def __getitem__(self, name):
        return self.find(name)


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
