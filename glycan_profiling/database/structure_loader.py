from collections import namedtuple
from functools import partial

from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycopeptidepy.structure.sequence import PeptideSequence
from glycopeptidepy.structure.parser import sequence_tokenizer
from glycopeptidepy.algorithm import reverse_preserve_sequon
from glycopeptidepy.structure.glycan import HashableGlycanComposition

from .lru import LRUCache


class GlycanCompositionCache(dict):
    pass


class CachingGlycanCompositionParser(object):
    def __init__(self, cache_size=4000):
        self.cache = GlycanCompositionCache()
        self.cache_size = cache_size
        self.lru = LRUCache()

    def _check_cache_valid(self):
        lru = self.lru
        while len(self.cache) > self.cache_size:
            key = lru.get_least_recently_used()
            lru.remove_node(key)
            value = self.cache.pop(key)
            try:
                value.clear_caches()
            except AttributeError:
                pass

    def _make_new_value(self, struct):
        value = HashableGlycanComposition.parse(struct.composition)
        value.id = struct.id
        return value

    def _populate_cache(self, struct, key):
        self._check_cache_valid()
        value = self._make_new_value(struct)
        self.cache[key] = value
        self.lru.add_node(key)
        return value

    def _extract_key(self, struct):
        return struct.composition

    def parse(self, struct):
        struct_key = self._extract_key(struct)
        try:
            seq = self.cache[struct_key]
            self.lru.hit_node(struct_key)
            return seq
        except KeyError:
            return self._populate_cache(struct, struct_key)

    def __call__(self, value):
        return self.parse(value)


hashable_glycan_glycopeptide_parser = partial(
    sequence_tokenizer, glycan_parser_function=HashableGlycanComposition.parse)


class GlycanFragmentCache(object):
    def __init__(self):
        self.cache = dict()

    def get_oxonium_ions(self, glycopeptide):
        try:
            return self.cache[glycopeptide.glycan]
        except:
            oxonium_ions = list(glycopeptide._glycan_fragments())
            self.cache[glycopeptide.glycan] = oxonium_ions
            return oxonium_ions

    def __call__(self, glycopeptide):
        return self.get_oxonium_ions(glycopeptide)


oxonium_ion_cache = GlycanFragmentCache()


class PeptideProteinRelation(SpanningMixin):
    __slots__ = ["start_position", "end_position", "protein_id", "hypothesis_id"]

    def __init__(self, start_position, end_position, protein_id, hypothesis_id):
        self.start_position = start_position
        self.end_position = end_position
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id

    @property
    def start(self):
        return self.start_position

    @property
    def end(self):
        return self.end_position

    def __repr__(self):
        return "PeptideProteinRelation(%d, %d, %d, %d)" % (self.start, self.end, self.protein_id, self.hypothesis_id)

    def __iter__(self):
        yield self.start_position
        yield self.end_position
        yield self.protein_id
        yield self.hypothesis_id

    def __reduce__(self):
        return self.__class__, tuple(self)

    def __eq__(self, other):
        return (self.start_position == other.start_position and
                self.end_position == other.end_position and
                self.protein_id == other.protein_id and
                self.hypothesis_id == other.hypothesis_id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.start_position, self.end_position))


class FragmentCachingGlycopeptide(PeptideSequence):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('parser_function', hashable_glycan_glycopeptide_parser)
        super(FragmentCachingGlycopeptide, self).__init__(*args, **kwargs)
        self.fragment_caches = {}
        self.protein_relation = None

    def __eq__(self, other):
        try:
            return (self.protein_relation == other.protein_relation) and (
                self.sequence == other.sequence)
        except AttributeError:
            return super(FragmentCachingGlycopeptide, self).__eq__(other)

    def get_fragments(self, *args, **kwargs):
        key = ("get_fragments", args, frozenset(kwargs.items()))
        try:
            return self.fragment_caches[key]
        except KeyError:
            result = list(super(FragmentCachingGlycopeptide, self).get_fragments(*args, **kwargs))
            self.fragment_caches[key] = result
            return result

    def stub_fragments(self, *args, **kwargs):
        key = ('stub_fragments', args, frozenset(kwargs.items()))
        try:
            return self.fragment_caches[key]
        except KeyError:
            result = list(super(FragmentCachingGlycopeptide, self).stub_fragments(*args, **kwargs))
            self.fragment_caches[key] = result
            return result

    def _glycan_fragments(self):
        return list(super(FragmentCachingGlycopeptide, self).glycan_fragments(oxonium=True))

    def glycan_fragments(self, *args, **kwargs):
        return oxonium_ion_cache(self)

    def clear_caches(self):
        self.fragment_caches.clear()

    def clone(self, *args, **kwargs):
        new = FragmentCachingGlycopeptide(str(self))
        try:
            new.id = self.id
        except AttributeError:
            pass
        # Intentionally share caches with offspring
        new.fragment_caches = self.fragment_caches
        return new

    def __repr__(self):
        return str(self)


KeyTuple = namedtuple("KeyTuple", ['id', 'sequence'])


class GlycopeptideCache(object):
    def __init__(self):
        self.sequence_map = dict()
        self.key_map = dict()

    def __getitem__(self, key):
        try:
            result = self.key_map[key]
            return result
        except KeyError:
            value = self.sequence_map[key.sequence]
            value = value.clone()
            self.key_map[key] = value
            return value

    def __setitem__(self, key, value):
        self.key_map[key] = value
        self.sequence_map[key.sequence] = value

    def __len__(self):
        return len(self.key_map)

    def pop(self, key):
        self.key_map.pop(key)
        self.sequence_map.pop(key.sequence, None)


class CachingGlycopeptideParser(object):
    def __init__(self, cache_size=4000):
        self.cache = GlycopeptideCache()
        self.cache_size = cache_size
        self.lru = LRUCache()
        self.churn = 0

    def _check_cache_valid(self):
        lru = self.lru
        while len(self.cache) > self.cache_size:
            self.churn += 1
            key = lru.get_least_recently_used()
            lru.remove_node(key)
            value = self.cache.pop(key)
            try:
                value.clear_caches()
            except AttributeError:
                pass

    def _make_new_value(self, struct):
        value = FragmentCachingGlycopeptide(struct.glycopeptide_sequence)
        value.id = struct.id
        value.protein_relation = PeptideProteinRelation(
            struct.start_position, struct.end_position,
            struct.protein_id, struct.hypothesis_id)
        return value

    def _populate_cache(self, struct, key):
        self._check_cache_valid()
        value = self._make_new_value(struct)
        self.cache[key] = value
        self.lru.add_node(key)
        return value

    def _extract_key(self, struct):
        return KeyTuple(struct.id, struct.glycopeptide_sequence)

    def parse(self, struct):
        struct_key = self._extract_key(struct)
        try:
            seq = self.cache[struct_key]
            self.lru.hit_node(struct_key)
            return seq
        except KeyError:
            return self._populate_cache(struct, struct_key)

    def __call__(self, value):
        return self.parse(value)


class DecoyMakingCachingGlycopeptideParser(CachingGlycopeptideParser):

    def _make_new_value(self, struct):
        value = FragmentCachingGlycopeptide(str(reverse_preserve_sequon(struct.glycopeptide_sequence)))
        value.id = struct.id
        value.protein_relation = PeptideProteinRelation(
            struct.start_position, struct.end_position,
            struct.protein_id, struct.hypothesis_id)
        return value
