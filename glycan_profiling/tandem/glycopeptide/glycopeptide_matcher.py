from collections import namedtuple
from functools import partial

from ..spectrum_matcher_base import TandemClusterEvaluatorBase
from ..chromatogram_mapping import ChromatogramMSMSMapper
from glycan_profiling.database import LRUCache
from glycresoft_sqlalchemy.structure.sequence import PeptideSequence, memoize
from glycresoft_sqlalchemy.structure.glycan import HashableGlycanComposition
from glycresoft_sqlalchemy.structure.parser import sequence_tokenizer
from .make_decoys import reverse_preserve_sequon
from .scoring import TargetDecoyAnalyzer


class GlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database, parser_type=None):
        if parser_type is None:
            parser_type = self._default_parser_type()
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.parser_type = parser_type
        self.parser = parser_type()

    def _default_parser_type(self):
        return CachingGlycopeptideParser

    def reset_parser(self):
        self.parser = self.parser_type()

    def evaluate(self, scan, structure, *args, **kwargs):
        target = self.parser(structure)
        matcher = self.scorer_type.evaluate(scan, target, *args, **kwargs)
        return matcher


class DecoyGlycopeptideMatcher(GlycopeptideMatcher):
    def _default_parser_type(self):
        return DecoyMakingCachingGlycopeptideParser


class TargetDecoyInterleavingGlycopeptideMatcher(TandemClusterEvaluatorBase):
    def __init__(self, tandem_cluster, scorer_type, structure_database):
        self.tandem_cluster = tandem_cluster
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.target_evaluator = GlycopeptideMatcher([], self.scorer_type, self.structure_database)
        self.decoy_evaluator = DecoyGlycopeptideMatcher([], self.scorer_type, self.structure_database)

    def score_one(self, scan, precursor_error_tolerance=1e-5, *args, **kwargs):
        target_result = self.target_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        decoy_result = self.decoy_evaluator.score_one(scan, precursor_error_tolerance, *args, **kwargs)
        return target_result, decoy_result

    def score_all(self, precursor_error_tolerance=1e-5, simplify=False, *args, **kwargs):
        target_out = []
        decoy_out = []
        for scan in self.tandem_cluster:
            target_result, decoy_result = self.score_one(scan, precursor_error_tolerance, *args, **kwargs)
            if len(target_result) > 0:
                target_out.append(target_result)
            if len(decoy_result) > 0:
                decoy_out.append(decoy_result)
        if simplify:
            for case in target_out:
                case.simplify()
                case.select_top()
            for case in decoy_out:
                case.simplify()
                case.select_top()
        return target_out, decoy_out


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


class FragmentCachingGlycopeptide(PeptideSequence):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('parser_function', hashable_glycan_glycopeptide_parser)
        super(FragmentCachingGlycopeptide, self).__init__(*args, **kwargs)
        self.fragment_caches = {}

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
        # Intentionally share caches with offspring
        new.fragment_caches = self.fragment_caches
        return new


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
        self.sequence_map.pop(key.sequence)


class CachingGlycopeptideParser(object):
    def __init__(self, cache_size=4000):
        self.cache = GlycopeptideCache()
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
        value = FragmentCachingGlycopeptide(struct.glycopeptide_sequence)
        value.id = struct.id
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
        return value


def chunkiter(collection, size=200):
    i = 0
    while collection[i:(i + size)]:
        yield collection[i:(i + size)]
        i += size


class GlycopeptideDatabaseSearchIdentifier(object):
    def __init__(self, tandem_scans, scorer_type, structure_database, scan_id_to_rt=lambda x: x):
        self.tandem_scans = sorted(
            tandem_scans, key=lambda x: x.precursor_information.extracted_neutral_mass, reverse=True)
        self.scorer_type = scorer_type
        self.structure_database = structure_database
        self.scan_id_to_rt = scan_id_to_rt

    def log(self, message):
        from datetime import datetime
        print(datetime.now().isoformat(' ') + ' ' + str(message))

    def search(self, precursor_error_tolerance=1e-5, simplify=True, chunk_size=200, limit=None, *args, **kwargs):
        target_hits = []
        decoy_hits = []
        total = len(self.tandem_scans)
        count = 0
        if limit is None:
            limit = float('inf')
        for bunch in chunkiter(self.tandem_scans, chunk_size):
            count += len(bunch)
            self.log("... Searching %s (%d/%d)" % (bunch[0].precursor_information, count, total))
            if hasattr(bunch[0], 'convert'):
                bunch = [o.convert(fitted=False, deconvoluted=True) for o in bunch]
            t, d = TargetDecoyInterleavingGlycopeptideMatcher(
                bunch, self.scorer_type, self.structure_database).score_all(
                precursor_error_tolerance=precursor_error_tolerance, simplify=simplify, *args, **kwargs)
            target_hits.extend(t)
            decoy_hits.extend(d)
            if count >= limit:
                self.log("Reached Limit. Halting.")
                break
        self.log('Search Done')
        return target_hits, decoy_hits

    def target_decoy(self, target_hits, decoy_hits, with_pit=False, *args, **kwargs):
        self.log("Running Target Decoy Analysis")
        tda = TargetDecoyAnalyzer(target_hits, decoy_hits, *args, with_pit=with_pit, **kwargs)
        tda.q_values()
        return target_hits

    def map_to_chromatograms(self, chromatograms, tandem_identifications, precursor_error_tolerance=1e-5):
        self.log("Mapping MS/MS Identifications onto Chromatograms")
        mapper = ChromatogramMSMSMapper(
            self.chromatograms, precursor_error_tolerance, self.scan_id_to_rt)
        mapper.assign_solutions_to_chromatograms(tandem_identifications)
        mapper.distribute_orphans()
        return mapper.chromatograms
