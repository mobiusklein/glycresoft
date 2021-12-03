from collections import namedtuple
from functools import partial

import numpy as np

from ms_deisotope.peak_dependency_network.intervals import SpanningMixin

from glycopeptidepy.structure.composition import Composition
from glycopeptidepy.structure.sequence import PeptideSequence
from glycopeptidepy.structure.parser import sequence_tokenizer
from glycopeptidepy.algorithm import reverse_preserve_sequon
from glycopeptidepy.structure.glycan import HashableGlycanComposition, GlycanCompositionWithOffsetProxy
from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy


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


class TextHashableGlycanCompositionParser(object):
    def __init__(self, size=int(2**16)):
        # self.cache = LRUMapping(size)
        self.cache = {}
        self.size = size

    def _parse(self, text):
        return HashableGlycanComposition.parse(text)

    def parse(self, text):
        try:
            return GlycanCompositionWithOffsetProxy(self.cache[text])
        except KeyError:
            inst = self._parse(text)
            if len(self.cache) > self.size and self.size != -1:
                self.cache.popitem()
            self.cache[text] = inst
            return GlycanCompositionWithOffsetProxy(inst)

    def __call__(self, text):
        return self.parse(text)


_glycan_parser = TextHashableGlycanCompositionParser()

hashable_glycan_glycopeptide_parser = partial(
    sequence_tokenizer, glycan_parser_function=_glycan_parser)


class GlycanFragmentCache(object):
    def __init__(self):
        self.cache = dict()

    def get_oxonium_ions(self, glycopeptide):
        key = str(glycopeptide.glycan)
        try:
            return self.cache[key]
        except KeyError:
            oxonium_ions = list(glycopeptide._glycan_fragments())
            self.cache[key] = oxonium_ions
            return oxonium_ions

    def __call__(self, glycopeptide):
        return self.get_oxonium_ions(glycopeptide)

    def update(self, source):
        if isinstance(source, dict):
            self.cache.update(source)
        else:
            self.cache.update(source.cache)

    def populate(self, glycan_composition_iterator):
        # A template peptide sequence which won't matter
        peptide = PeptideSequence("PEPTIDE")
        # Pretend to support the backdoor method
        peptide._glycan_fragments = peptide.glycan_fragments
        # Attach each glycan composition to the peptide and
        # calculate oxonium ions and cache them ahead of time.
        for gc in glycan_composition_iterator:
            gc = gc.clone()
            peptide.glycan = gc
            self(peptide)


oxonium_ion_cache = GlycanFragmentCache()


class PeptideProteinRelation(SpanningMixin):
    __slots__ = ["protein_id", "hypothesis_id"]

    def __init__(self, start_position, end_position, protein_id, hypothesis_id):  # pylint: disable=super-init-not-called
        self.start = start_position
        self.end = end_position
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id

    @property
    def start_position(self):
        return int(self.start)

    @start_position.setter
    def start_position(self, value):
        self.start = value

    @property
    def end_position(self):
        return int(self.end)

    @end_position.setter
    def end_position(self, value):
        self.end = value

    def __repr__(self):
        return "PeptideProteinRelation(%d, %d, %r, %r)" % (
            self.start_position, self.end_position, self.protein_id, self.hypothesis_id)

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


class NamedPeptideProteinRelation(PeptideProteinRelation):
    __slots__ = ("protein_name", )

    def __init__(self, start_position, end_position, protein_id, hypothesis_id, protein_name=None):
        super(NamedPeptideProteinRelation, self).__init__(
            start_position, end_position, protein_id, hypothesis_id)
        self.protein_name = protein_name

    def __iter__(self):
        yield self.start_position
        yield self.end_position
        yield self.protein_id
        yield self.hypothesis_id
        yield self.protein_name

    def __reduce__(self):
        return self.__class__, tuple(self)

    def __eq__(self, other):
        coords =  (self.start_position == other.start_position and
                   self.end_position == other.end_position)
        if coords:
            if self.protein_name is not None:
                try:
                    coords = self.protein_name == other.protein_name
                except AttributeError:
                    coords = self.protein_id == other.protein_id
            else:
                coords = self.protein_id == other.protein_id
        return coords


# Add a new default attribute value to the parent class so all future instances
# of the parent class (and all sub-classes) have a fallback value.
PeptideSequence.glycan_prior = 0.0


class GlycopeptideFragmentCachingContext(object):
    __slots__ = ('store', )

    def __init__(self, store=None):
        if store is None:
            store = {}
        self.store = store

    def peptide_backbone_fragment_key(self, target, args, kwargs):
        key = ("get_fragments", args, frozenset(kwargs.items()))
        return key

    def stub_fragment_key(self, target, args, kwargs):
        key = ('stub_fragments', args, frozenset(kwargs.items()), )
        return key

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def clear(self):
        self.store.clear()

    def bind(self, target):
        target.fragment_caches = self
        return target

    def unbind(self, target):
        target.fragment_caches = self.__class__()
        return target

    def __call__(self, target):
        return self.bind(target)

    def _make_target_key(self, key):
        # value = key[-1]
        # as_target_peptide = StructureClassification[int(value) ^ 1]
        # new_key = key[:-1] + (as_target_peptide, )
        # return new_key
        return None


try:
    from glycan_profiling._c.structure.structure_loader import peptide_backbone_fragment_key
    GlycopeptideFragmentCachingContext.peptide_backbone_fragment_key = peptide_backbone_fragment_key
except ImportError:
    pass


class GlycanAwareGlycopeptideFragmentCachingContext(GlycopeptideFragmentCachingContext):
    def stub_fragment_key(self, target, args, kwargs):
        tid = target.id
        key = ('stub_fragments', args, frozenset(
            kwargs.items()), tid.glycan_combination_id, tid.structure_type)
        return key

    def _make_target_key(self, key):
        from glycan_profiling.tandem.glycopeptide.dynamic_generation.search_space import StructureClassification
        value = key[-1]
        as_target_peptide = StructureClassification[int(value) ^ 1]
        new_key = key[:-1] + (as_target_peptide, )
        return new_key

try:
    from glycan_profiling._c.structure.structure_loader import GlycopeptideFragmentCachingContext, GlycanAwareGlycopeptideFragmentCachingContext
except ImportError as err:
    print(err)
    pass


class FragmentCachingGlycopeptide(PeptideSequence):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('parser_function', hashable_glycan_glycopeptide_parser)
        super(FragmentCachingGlycopeptide, self).__init__(*args, **kwargs)
        self.fragment_caches = GlycopeptideFragmentCachingContext()
        self.protein_relation = None
        self.id = None
        self.glycan_prior = 0.0

    def __reduce__(self):
        return self.__class__, (str(self), ), self.__getstate__()

    def __getstate__(self):
        state = {}
        state['protein_relation'] = self.protein_relation
        state['id'] = self.id
        state['glycan_prior'] = self.glycan_prior
        return state

    def __setstate__(self, state):
        self.protein_relation = state['protein_relation']
        self.id = state['id']
        self.glycan_prior = state.get('glycan_prior', 0.0)

    def __eq__(self, other):
        try:
            return (self.protein_relation == other.protein_relation) and (
                super(FragmentCachingGlycopeptide, self).__eq__(other))
        except AttributeError:
            return super(FragmentCachingGlycopeptide, self).__eq__(other)

    __hash__ = PeptideSequence.__hash__

    def __ne__(self, other):
        return not self == other

    def get_fragments(self, *args, **kwargs):  # pylint: disable=arguments-differ
        key = self.fragment_caches.peptide_backbone_fragment_key(self, args, kwargs)
        try:
            return self.fragment_caches[key]
        except KeyError:
            result = list(super(FragmentCachingGlycopeptide, self).get_fragments(*args, **kwargs))
            self.fragment_caches[key] = result
            return result

    def stub_fragments(self, *args, **kwargs):  # pylint: disable=arguments-differ
        kwargs.setdefault("strategy", CachingStubGlycopeptideStrategy)
        key = self.fragment_caches.stub_fragment_key(self, args, kwargs)
        try:
            return self.fragment_caches[key]
        except KeyError:
            result = list(super(FragmentCachingGlycopeptide, self).stub_fragments(*args, **kwargs))
            self.fragment_caches[key] = result
            return result

    def _glycan_fragments(self):
        return list(super(FragmentCachingGlycopeptide, self).glycan_fragments(oxonium=True))

    def glycan_fragments(self, *args, **kwargs):  # pylint: disable=arguments-differ
        return oxonium_ion_cache(self)

    def clear_caches(self):
        self.fragment_caches = GlycopeptideFragmentCachingContext()

    def clone(self, *args, **kwargs):  # pylint: disable=arguments-differ
        share_cache = kwargs.pop("share_cache", True)
        new = super(FragmentCachingGlycopeptide, self).clone(*args, **kwargs)
        new.id = self.id
        new.protein_relation = self.protein_relation
        # Intentionally share caches with offspring
        if share_cache:
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
    __slots__ = ('cache', 'cache_size', 'lru', 'churn', 'sequence_cls')

    def __init__(self, cache_size=4000, sequence_cls=FragmentCachingGlycopeptide):
        self.cache = GlycopeptideCache()
        self.cache_size = cache_size
        self.lru = LRUCache()
        self.churn = 0
        self.sequence_cls = sequence_cls

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
        value = self.sequence_cls(struct.glycopeptide_sequence)
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


class CachingPeptideParser(CachingGlycopeptideParser):
    __slots__ = ()

    def _make_new_value(self, struct):
        value = self.sequence_cls(struct.modified_peptide_sequence)
        value.id = struct.id
        value.protein_relation = PeptideProteinRelation(
            struct.start_position, struct.end_position,
            struct.protein_id, struct.hypothesis_id)
        return value

    def _extract_key(self, struct):
        return KeyTuple(struct.id, struct.modified_peptide_sequence)


class DecoyFragmentCachingGlycopeptide(FragmentCachingGlycopeptide):
    _random_shift_cache = dict()

    @classmethod
    def get_random_shifts_for(cls, mass, n, low=1.0, high=30.0):
        seed = int(round(mass))
        key = (seed, n)
        try:
            return cls._random_shift_cache[key]
        except KeyError:
            rng = np.random.RandomState(seed)
            rand_deltas = rng.uniform(low, high, n)
            cls._random_shift_cache[key] = rand_deltas
            return rand_deltas

    def _permute_stub_masses(self, stub_fragments, kwargs, do_clone=False):
        random_low = kwargs.get('random_low', 1.0)
        random_high = kwargs.get("random_high", 30.0)
        n = len(stub_fragments)
        rand_deltas = self.get_random_shifts_for(
            self.glycan_composition.mass(),
            n, random_low, random_high)
        i = 0
        stub_fragments = self._clone_and_shift_stub_fragments(
            stub_fragments, rand_deltas, do_clone)
        return stub_fragments

    @staticmethod
    def _clone_and_shift_stub_fragments(stubs, rand_deltas, do_clone=True):
        i = 0
        if do_clone:
            result = []
        for frag in stubs:
            if do_clone:
                frag = frag.clone()
            if frag.glycosylation_size > 1:
                delta = rand_deltas[i]
                i += 1
                frag.mass += delta
            if do_clone:
                result.append(frag)
        if do_clone:
            return result
        return stubs

    def stub_fragments(self, *args, **kwargs):
        kwargs.setdefault("strategy", CachingStubGlycopeptideStrategy)
        key = self.fragment_caches.stub_fragment_key(self, args, kwargs)
        try:
            return self.fragment_caches[key]
        except KeyError:
            result = list(
                # Directly call the superclass method of FragmentCachingGlycopeptide as we
                # do not need to go through a preliminary round of cache key construction and
                # querying.
                super(FragmentCachingGlycopeptide, self).stub_fragments( # pylint: disable=bad-super-call
                    *args, **kwargs))
            result = self._permute_stub_masses(result, kwargs)
            self.fragment_caches[key] = result
            return result

    @classmethod
    def from_target(cls, target):
        inst = cls()
        if target._glycosylation_manager.aggregate is not None:
            glycan = target._glycosylation_manager.aggregate.clone()
            glycan.composition_offset = Composition("H2O")
        else:
            glycan = None
        inst._init_from_components(
            target.sequence, glycan,
            target.n_term.modification,
            target.c_term.modification)
        try:
            inst.id = target.id
            inst.protein_relation = target.protein_relation
        except AttributeError:
            inst.protein_relation = None
        # Intentionally share caches with offspring
        inst.fragment_caches = inst.fragment_caches.__class__(
            {k: v for k, v in target.fragment_caches.items() if 'stub_fragments' not in k})
        return inst


try:
    from glycan_profiling._c.structure.structure_loader import clone_and_shift_stub_fragments as _clone_and_shift_stub_fragments
    DecoyFragmentCachingGlycopeptide._clone_and_shift_stub_fragments = staticmethod(
        _clone_and_shift_stub_fragments)
except ImportError:
    pass


class CachingStubGlycopeptideStrategy(StubGlycopeptideStrategy):
    _cache = dict()

    def n_glycan_composition_fragments(self, glycan, core_count=1, iteration_count=0):
        key = (str(glycan), core_count, iteration_count, self.extended, self.extended_fucosylation)
        try:
            value = self._cache[key]
            return value
        except KeyError:
            value = super(CachingStubGlycopeptideStrategy, self).n_glycan_composition_fragments(
                glycan, core_count, iteration_count)
            self._cache[key] = value
            return value

    @classmethod
    def update(cls, source):
        cls._cache.update(source)

    @classmethod
    def populate(cls, glycan_composition_iterator, **kwargs):
        inst = cls(None, **kwargs)
        for gc in glycan_composition_iterator:
            gc = gc.clone()
            inst.n_glycan_composition_fragments(gc, 1, 0)

    @classmethod
    def get_cache(cls):
        return cls._cache


class SequenceReversingCachingGlycopeptideParser(CachingGlycopeptideParser):
    __slots__ = ()

    def _make_new_value(self, struct):
        value = self.sequence_cls(str(reverse_preserve_sequon(struct.glycopeptide_sequence)))
        value.id = struct.id
        value.protein_relation = PeptideProteinRelation(
            struct.start_position, struct.end_position,
            struct.protein_id, struct.hypothesis_id)
        return value


class GlycopeptideDatabaseRecord(object):
    __slots__ = [
        "id", "calculated_mass",
        "glycopeptide_sequence",
        "protein_id",
        "start_position",
        "end_position",
        "peptide_mass",
        "hypothesis_id"
    ]

    def __init__(self, id, calculated_mass, glycopeptide_sequence, protein_id,
                 start_position, end_position, peptide_mass, hypothesis_id):
        self.id = id
        self.calculated_mass = calculated_mass
        self.glycopeptide_sequence = glycopeptide_sequence
        self.protein_id = protein_id
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_mass = peptide_mass
        self.hypothesis_id = hypothesis_id

    def __reduce__(self):
        return self.__class__, (self.id, self.calculated_mass, self.glycopeptide_sequence, self.protein_id,
                                self.start_position, self.end_position, self.peptide_mass, self.hypothesis_id)

    def __repr__(self):
        template = (
            "{self.__class__.__name__}(id={self.id}, calculated_mass={self.calculated_mass}, "
            "glycopeptide_sequence={self.glycopeptide_sequence}, protein_id={self.protein_id}, "
            "start_position={self.start_position}, end_position={self.end_position}, "
            "peptide_mass={self.peptide_mass}, hypothesis_id={self.hypothesis_id}, ")
        return template.format(self=self)


class PeptideDatabaseRecordBase(object):
    __slots__ = ['id', "calculated_mass", "modified_peptide_sequence", "protein_id", "start_position", "end_position",
                 "hypothesis_id", "n_glycosylation_sites", "o_glycosylation_sites", "gagylation_sites"]

    def __hash__(self):
        return hash(self.modified_peptide_sequence)

    def __eq__(self, other):
        if other is None:
            return False
        if self.id != other.id:
            return False
        if self.protein_id != other.protein_id:
            return False
        if abs(self.calculated_mass - other.calculated_mass) > 1e-3:
            return False
        if self.start_position != other.start_position:
            return False
        if self.end_position != other.end_position:
            return False
        if self.hypothesis_id != other.hypothesis_id:
            return False
        if self.n_glycosylation_sites != other.n_glycosylation_sites:
            return False
        if self.o_glycosylation_sites != other.o_glycosylation_sites:
            return False
        if self.gagylation_sites != other.gagylation_sites:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def has_glycosylation_sites(self):
        return (len(self.n_glycosylation_sites) + len(self.o_glycosylation_sites) + len(self.gagylation_sites)) > 0

    @classmethod
    def from_record(cls, record):
        return cls(**record)


try:
    from glycan_profiling._c.structure.structure_loader import PeptideDatabaseRecordBase
except ImportError:
    pass

class PeptideDatabaseRecord(PeptideDatabaseRecordBase):
    __slots__ = ()

    def __init__(self, id, calculated_mass, modified_peptide_sequence, protein_id, start_position, end_position,
                 hypothesis_id, n_glycosylation_sites, o_glycosylation_sites, gagylation_sites):
        self.id = id
        self.calculated_mass = calculated_mass
        self.modified_peptide_sequence = modified_peptide_sequence
        self.protein_id = protein_id
        self.start_position = start_position
        self.end_position = end_position
        self.hypothesis_id = hypothesis_id
        self.n_glycosylation_sites = tuple(n_glycosylation_sites)
        self.o_glycosylation_sites = tuple(o_glycosylation_sites)
        self.gagylation_sites = tuple(gagylation_sites)

    def convert(self):
        peptide = FragmentCachingGlycopeptide(self.modified_peptide_sequence)
        peptide.id = self.id
        rel = PeptideProteinRelation(
            self.start_position, self.end_position, self.protein_id, self.hypothesis_id)
        peptide.protein_relation = rel
        return peptide

    def __repr__(self):
        fields = ', '.join(["%s=%r" % (n, getattr(self, n)) for n in [
            'id', "calculated_mass", "modified_peptide_sequence", "protein_id", "start_position", "end_position",
            "hypothesis_id", "n_glycosylation_sites", "o_glycosylation_sites", "gagylation_sites"]])
        return "{self.__class__.__name__}({fields})".format(self=self, fields=fields)


class LazyGlycopeptide(object):
    __slots__ = ("sequence", "id", "protein_relation")

    def __init__(self, sequence, id, protein_relation=None):
        self.sequence = sequence
        self.id = id
        self.protein_relation = protein_relation

    def convert(self, sequence_cls=None):
        if sequence_cls is None:
            sequence_cls = FragmentCachingGlycopeptide
        inst = sequence_cls(self.sequence)
        inst.id = self.id
        inst.protein_relation = self.protein_relation
        return inst

    def __iter__(self):
        for pos in self.convert():
            yield pos

    def __getitem__(self, i):
        return self.convert()[i]

    def __len__(self):
        result = hashable_glycan_glycopeptide_parser(self.sequence)
        return len(result[0])

    @property
    def glycan_composition(self):
        result = hashable_glycan_glycopeptide_parser(self.sequence)
        glycan_composition = result[2]
        return glycan_composition

    def __repr__(self):
        return "{self.__class__.__name__}({self.sequence}, {self.id})".format(self=self)

    def __str__(self):
        return str(self.sequence)

    def __eq__(self, other):
        return self.sequence == other.sequence and self.id == other.id

    def __hash__(self):
        return hash(self.sequence)


class GlycanCompositionDeltaCache(object):
    def __init__(self, storage=None):
        if storage is None:
            storage = {}
        self.storage = storage

    def __call__(self, x, y):
        key = (x, y)
        try:
            return self.storage[key]
        except KeyError:
            delta = (x - y)
            self.storage[key] = delta
            return delta
