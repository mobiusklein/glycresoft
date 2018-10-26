'''Auxiliary data structures used to look up records from a database
using the Sequence interface or the Mapping interface for convenience.

Maintainence Note: Mostly unused outside of debugging
'''

from glycan_profiling.serialize import (
    Protein, Peptide, Glycopeptide)

from glycan_profiling.structure import LRUCache


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


class GlycopeptideSequenceCache(object):
    def __init__(self, session, cache_size=1e6):
        self.session = session
        self.cache_size = int(cache_size)
        self.cache = dict()
        self.lru = LRUCache()

    def _fetch_sequence_by_id(self, id):
        return self.session.query(Glycopeptide.glycopeptide_sequence).filter(
            Glycopeptide.id == id).scalar()

    def _check_cache_valid(self):
        lru = self.lru
        while len(self.cache) > self.cache_size:
            self.churn += 1
            key = lru.get_least_recently_used()
            lru.remove_node(key)
            self.cache.pop(key)

    def _populate_cache(self, id):
        self._check_cache_valid()
        value = self._fetch_sequence_by_id(id)
        self.cache[id] = value
        self.lru.add_node(id)
        return value

    def _get_sequence_by_id(self, id):
        try:
            return self.cache[id]
        except KeyError:
            return self._populate_cache(id)

    def __getitem__(self, key):
        return self._get_sequence_by_id(id)

    def _fetch_batch(self, ids, chunk_size=500):
        n = len(ids)
        i = 0
        acc = []
        while i < n:
            batch = ids[i:(i + chunk_size)]
            i += chunk_size
            seqs = self.session.query(Glycopeptide.glycopeptide_sequence).filter(
                Glycopeptide.id.in_(batch)).all()
            acc.extend(s[0] for s in seqs)
        return acc

    def _process_batch(self, ids, chunk_size=500):
        result = dict()
        missing = []
        for i in ids:
            try:
                result[i] = self.cache[i]
            except KeyError:
                missing.append(i)
        fetched = self._fetch_batch(missing, chunk_size)
        for i, v in zip(missing, fetched):
            self.cache[i] = v
            result[i] = v
        return result

    def batch(self, ids, chunk_size=500):
        self._check_cache_valid()
        return self._process_batch(ids, chunk_size)


class GlycopeptideBatchManager(object):
    def __init__(self, cache):
        self.cache = cache
        self.batch = {}

    def mark_hit(self, match):
        self.batch[match.id] = match
        return match

    def process_batch(self):
        ids = [m for m, v in self.batch.items() if v.glycopeptide_sequence is None]
        seqs = self.cache.batch(ids)
        for k, v in seqs.items():
            self.batch[k].glycopeptide_sequence = v

    def clear(self):
        self.batch.clear()
