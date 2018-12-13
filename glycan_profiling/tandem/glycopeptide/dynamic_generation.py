import operator
import logging
import struct

from collections import namedtuple

from glypy.utils import Enum
from glycopeptidepy import GlycosylationType
from glycopeptidepy.structure.sequence import (
    _n_glycosylation, _o_glycosylation, _gag_linker_glycosylation)

from ms_deisotope import isotopic_shift

from glycan_profiling.chromatogram_tree import Unmodified

from glycan_profiling.structure.lru import LRUMapping
from glycan_profiling.structure import (FragmentCachingGlycopeptide, DecoyFragmentCachingGlycopeptide)

from glycan_profiling.database import intervals
from glycan_profiling.database.mass_collection import NeutralMassDatabase, SearchableMassCollection
from glycan_profiling.database.builder.glycopeptide.common import limiting_combinations

from .core_search import GlycanCombinationRecord, GlycanFilteringPeptideMassEstimator
from .scoring import CoverageWeightedBinomialScorer
from ..workload import WorkloadManager


logger = logging.Logger("glycresoft.dynamic_generation")


_glycopeptide_key_t = namedtuple(
    'glycopeptide_key', ('start_position', 'end_position', 'peptide_id', 'protein_id',
                         'hypothesis_id', 'glycan_combination_id', 'structure_type'))


class StructureClassification(Enum):
    target_peptide_target_glycan = 0
    target_peptide_decoy_glycan = 1
    decoy_peptide_target_glycan = 2
    decoy_peptide_decoy_glycan = decoy_peptide_target_glycan | target_peptide_decoy_glycan


TT = StructureClassification.target_peptide_target_glycan


class glycopeptide_key_t(_glycopeptide_key_t):
    __slots__ = ()

    struct_spec = struct.Struct('!LLLLLLL')

    def serialize(self):
        return self.struct_spec.pack(*self)

    @classmethod
    def parse(cls, binary):
        return cls(*cls.struct_spec.unpack(binary))

    def copy(self, structure_type=None):
        if structure_type is None:
            structure_type = self.structure_type
        return self._replace(structure_type=structure_type)

    def to_decoy_glycan(self):
        structure_type = self.structure_type
        if structure_type == StructureClassification.target_peptide_target_glycan:
            structure_type = StructureClassification.target_peptide_decoy_glycan
        elif structure_type == StructureClassification.decoy_peptide_target_glycan:
            structure_type = StructureClassification.decoy_peptide_decoy_glycan
        return self.copy(structure_type)


class GlycoformGeneratorBase(object):
    @classmethod
    def from_hypothesis(cls, session, hypothesis_id):
        glycan_combinations = GlycanCombinationRecord.from_hypothesis(session, hypothesis_id)
        return cls(glycan_combinations)

    def __init__(self, glycan_combinations, cache_size=None, *args, **kwargs):
        if not isinstance(glycan_combinations, SearchableMassCollection):
            glycan_combinations = NeutralMassDatabase(
                list(glycan_combinations), operator.attrgetter("dehydrated_mass"))
        if cache_size is None:
            cache_size = 2 ** 15
        else:
            cache_size = int(cache_size)
        self.glycan_combinations = glycan_combinations
        self._peptide_cache = LRUMapping(cache_size)
        self._cache_hit = 0
        self._cache_miss = 0
        super(GlycoformGeneratorBase, self).__init__(*args, **kwargs)

    def handle_glycan_combination(self, peptide_obj, peptide_record, glycan_combination,
                                  glycosylation_sites, core_type):
        key = self._make_key(peptide_record, glycan_combination)
        if key in self._peptide_cache:
            self._cache_hit += 1
            return self._peptide_cache[key]
        self._cache_miss += 1
        glycosylation_sites_unoccupied = set(glycosylation_sites)
        for site in list(glycosylation_sites_unoccupied):
            if peptide_obj[site][1]:
                glycosylation_sites_unoccupied.remove(site)
        site_combinations = list(limiting_combinations(glycosylation_sites_unoccupied, glycan_combination.count))
        result_set = [None for i in site_combinations]
        for i, site_set in enumerate(site_combinations):
            glycoform = peptide_obj.clone(share_cache=False)
            glycoform.id = key
            glycoform.glycan = glycan_combination.composition.clone()
            for site in site_set:
                glycoform.add_modification(site, core_type.name)
            result_set[i] = glycoform
        result_set = Record.build(result_set)
        self._peptide_cache[key] = result_set
        return result_set

    def _make_key(self, peptide_record, glycan_combination, structure_type=TT):
        key = glycopeptide_key_t(
            peptide_record.start_position,
            peptide_record.end_position,
            peptide_record.id,
            peptide_record.protein_id,
            peptide_record.hypothesis_id,
            glycan_combination.id,
            structure_type)
        return key

    def handle_n_glycan(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.n_glycosylation_sites,
            _n_glycosylation)

    def handle_o_glycan(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.o_glycosylation_sites,
            _o_glycosylation)

    def handle_gag_linker(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj, peptide_record, glycan_combination, peptide_record.gagylation_sites,
            _gag_linker_glycosylation)

    def reset(self, **kwargs):
        self._cache_hit = 0
        self._cache_miss = 0
        self._peptide_cache.clear()


class PeptideGlycosylator(GlycoformGeneratorBase):
    def __init__(self, peptide_records, glycan_combinations, cache_size=2**16, *args, **kwargs):
        super(PeptideGlycosylator, self).__init__(glycan_combinations, *args, **kwargs)
        if not isinstance(peptide_records, SearchableMassCollection):
            peptide_records = NeutralMassDatabase(peptide_records)
        self.peptides = peptide_records

    def handle_peptide_mass(self, peptide_mass, intact_mass, error_tolerance=1e-5):
        peptide_records = self.peptides.search_mass_ppm(peptide_mass, error_tolerance)
        glycan_mass = intact_mass - peptide_mass
        glycan_combinations = self.glycan_combinations.search_mass_ppm(glycan_mass, error_tolerance)
        result_set = []
        for peptide in peptide_records:
            self._combinate(peptide, glycan_combinations, result_set)
        return result_set

    def _combinate(self, peptide, glycan_combinations, result_set=None):
        if result_set is None:
            result_set = []
        peptide_obj = peptide.convert()
        for glycan_combination in glycan_combinations:
            for tp in glycan_combination.glycan_types:
                tp = GlycosylationType[tp]
                if tp is GlycosylationType.n_linked:
                    result_set.extend(
                        self.handle_n_glycan(peptide_obj.clone(), peptide, glycan_combination))
                elif tp is GlycosylationType.o_linked:
                    result_set.extend(
                        self.handle_o_glycan(peptide_obj.clone(), peptide, glycan_combination))
                elif tp is GlycosylationType.glycosaminoglycan:
                    result_set.extend(
                        self.handle_gag_linker(peptide_obj.clone(), peptide, glycan_combination))
        return result_set

    def generate_crossproduct(self, lower_bound=0, upper_bound=float('inf')):
        minimum_peptide_mass = max(lower_bound - self.glycan_combinations.highest_mass, 0)
        maximum_peptide_mass = max(upper_bound - self.glycan_combinations.lowest_mass, 0)
        peptides = self.peptides.search_between(max(minimum_peptide_mass - 10, 0), maximum_peptide_mass)
        for peptide in peptides:
            if not peptide.has_glycosylation_sites():
                continue
            glycan_mass_limit = upper_bound - peptide.calculated_mass
            if glycan_mass_limit < 0:
                continue
            minimum_glycan_mass = max(lower_bound - peptide.calculated_mass, 0)
            glycan_combinations = self.glycan_combinations.search_between(minimum_glycan_mass, glycan_mass_limit + 10)
            for solution in self._combinate(peptide, glycan_combinations):
                total_mass = solution.total_mass
                if total_mass < lower_bound or total_mass > upper_bound:
                    continue
                yield solution

    def reset(self, **kwargs):
        super(PeptideGlycosylator, self).reset(**kwargs)
        self.peptides.reset(**kwargs)


class DynamicGlycopeptideSearchBase(object):
    neutron_shift = isotopic_shift()

    def handle_scan_group(self, group, precursor_error_tolerance=1e-5, mass_shifts=None):
        raise NotImplementedError()

    def reset(self):
        self.peptide_glycosylator.reset()


class PredictiveGlycopeptideSearch(DynamicGlycopeptideSearchBase):

    def __init__(self, peptide_glycosylator, product_error_tolerance=2e-5, coarse_threshold=0.1,
                 min_fragments=2, peptide_masses_per_scan=100,
                 probing_range_for_missing_precursors=3, trust_precursor_fits=True,
                 scorer_type=CoverageWeightedBinomialScorer):
        self.peptide_glycosylator = peptide_glycosylator
        self.product_error_tolerance = product_error_tolerance
        self.coarse_threshold = coarse_threshold
        self.min_fragments = min_fragments
        self.peptide_mass_predictor = GlycanFilteringPeptideMassEstimator(
            self.peptide_glycosylator.glycan_combinations,
            product_error_tolerance)
        self.peptide_masses_per_scan = peptide_masses_per_scan
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits
        self.scorer_type = scorer_type

    def handle_scan_group(self, group, precursor_error_tolerance=1e-5, mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        workload = WorkloadManager()

        coarse_threshold = self.coarse_threshold
        min_fragments = self.min_fragments
        estimate_peptide_mass = self.peptide_mass_predictor.estimate_peptide_mass
        handle_peptide_mass = self.peptide_glycosylator.handle_peptide_mass
        peptide_masses_per_scan = self.peptide_masses_per_scan
        for scan in group:
            workload.add_scan(scan)
            if (not scan.precursor_information.defaulted and self.trust_precursor_fits):
                probe = 0
            else:
                probe = self.probing_range_for_missing_precursors
            for i in range(probe + 1):
                neutron_shift = self.neutron_shift * i
                for mass_shift in mass_shifts:
                    seen = set()
                    mass_shift_name = mass_shift.name
                    intact_mass = scan.precursor_information.neutral_mass - mass_shift.mass - neutron_shift
                    for peptide_mass in estimate_peptide_mass(scan, topn=peptide_masses_per_scan, mass_shift=mass_shift,
                                                              threshold=coarse_threshold, min_fragments=min_fragments):
                        for candidate in handle_peptide_mass(peptide_mass, intact_mass):
                            key = (candidate.id, mass_shift_name)
                            if key in seen:
                                continue
                            seen.add(key)
                            workload.add_scan_hit(scan, candidate, mass_shift_name)
        return workload


DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT = 1e5


class IterativeGlycopeptideSearch(DynamicGlycopeptideSearchBase):
    total_mass_getter = operator.attrgetter('total_mass')

    def __init__(self, peptide_glycosylator, product_error_tolerance=2e-5, cache_size=2,
                 scorer_type=CoverageWeightedBinomialScorer,
                 threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT):
        self.peptide_glycosylator = peptide_glycosylator
        self.product_error_tolerance = product_error_tolerance
        self.scorer_type = scorer_type
        self.interval_cache = intervals.LRUIntervalSet(max_size=cache_size)
        self.threshold_cache_total_count = threshold_cache_total_count

    def _make_new_glycopeptides_for_interval(self, lower_bound, upper_bound):
        generator = self.peptide_glycosylator.generate_crossproduct(lower_bound, upper_bound)
        generator = NeutralMassDatabase(list(generator), self.total_mass_getter)
        return intervals.ConcatenateMassIntervalNode(generator)

    def _upkeep_memory_intervals(self, lower_bound=0, upper_bound=float('inf')):
        """Perform routine maintainence of the interval cache, ensuring its size does not
        exceed the upper limit
        """
        n = len(self.interval_cache)
        if n > 1:
            while (len(self.interval_cache) > 1 and
                   self.interval_cache.total_count >
                   self.threshold_cache_total_count):
                logger.info("Upkeep Memory Intervals %d %d", self.interval_cache.total_count, len(self.interval_cache))
                self.interval_cache.remove_lru_interval()
        n = len(self.interval_cache)
        if n == 1 and self.interval_cache.total_count > self.threshold_cache_total_count:
            segment = self.interval_cache[0]
            segment.constrain(lower_bound, upper_bound)

    def get_glycopeptides(self, lower_bound, upper_bound):
        self._upkeep_memory_intervals(lower_bound, upper_bound)
        q = intervals.QueryIntervalBase(
            (lower_bound + upper_bound) / 2., lower_bound, upper_bound)
        match = self.interval_cache.find_interval(q)
        if match is not None:
            logger.debug("Nearest interval %r", match)
            # We are completely contained in an existing interval, so just
            # use that one.
            if match.contains_interval(q):
                logger.debug("Query interval %r was completely contained in %r", q, match)
                return match.group
            # We overlap with an extending interval, so we should populate
            # the new one and merge them.
            elif match.overlaps(q):
                q2 = q.difference(match)
                logger.debug("Query interval partially overlapped, creating disjoint interval %r", q2)
                match = self.interval_cache.extend_interval(
                    match,
                    self._make_new_glycopeptides_for_interval(q2.start, q2.end)
                )
                return match.group
            # We might need to insert a new interval
            else:
                logger.debug("Query interval %r did not overlap with %r", q, match)
                return self._insert_interval(q.start, q.end)
        else:
            logger.debug("Query interval %r did not overlap with %r", q, match)
            return self._insert_interval(q.start, q.end)

    def _insert_interval(self, lower_bound, upper_bound):
        node = self._make_new_glycopeptides_for_interval(lower_bound, upper_bound)
        logger.debug("New Node: %r", node)
        # We won't insert this node if it is empty.
        if len(node.group) == 0:
            return node.group
        nearest_interval = self.interval_cache.find_interval(node)
        # Should an insert be performed if the query just didn't overlap well
        # with the database?
        if nearest_interval is None:
            # No nearby interval, so we should insert
            logger.debug("No nearby interval for %r", node)
            self.interval_cache.insert_interval(node)
            return node.group
        elif nearest_interval.overlaps(node):
            logger.debug("Nearest interval %r overlaps this %r", nearest_interval, node)
            nearest_interval = self.interval_cache.extend_interval(nearest_interval, node)
            return nearest_interval.group
        elif not nearest_interval.contains_interval(node):
            logger.debug("Nearest interval %r didn't contain this %r", nearest_interval, node)
            # Nearby interval didn't contain this interval
            self.interval_cache.insert_interval(node)
            return node.group
        else:
            # Situation unclear.
            # Not worth inserting, so just return the group
            logger.info("Unknown Condition Overlap %r / %r" % (node, nearest_interval))
            return nearest_interval.group

    def handle_scan_group(self, group, precursor_error_tolerance=1e-5, mass_shifts=None):
        if mass_shifts is None:
            mass_shifts = [Unmodified]

        group = NeutralMassDatabase(group, lambda x: x.precursor_information.neutral_mass)
        lo_mass = group.lowest_mass
        hi_mass = group.highest_mass
        shifts = ([m.mass for m in mass_shifts])

        lo_shift = min(shifts) - 0.5
        hi_shift = max(shifts) + 0.5

        lower_bound = lo_mass + lo_shift
        lower_bound = lower_bound - (lower_bound * precursor_error_tolerance)

        upper_bound = hi_mass + hi_shift
        upper_bound = upper_bound + (upper_bound * precursor_error_tolerance)

        workload = WorkloadManager()
        id_to_scan = {}
        for scan in group:
            id_to_scan[scan.id] = scan
        logger.info("\nQuerying between %f and %f", lower_bound, upper_bound)
        candidates = self.get_glycopeptides(lower_bound, upper_bound)
        logger.info("Interval from %f to %f contained %d candidates", lower_bound, upper_bound, len(candidates))
        for scan in group:
            for mass_shift in mass_shifts:
                intact_mass = scan.precursor_information.neutral_mass - mass_shift.mass
                for candidate in candidates.search_mass_ppm(intact_mass, precursor_error_tolerance):
                    workload.add_scan_hit(scan, candidate, mass_shift.name)
        return workload


class Record(object):
    __slots__ = ('glycopeptide', 'id', 'total_mass')

    def __init__(self, glycopeptide=None):
        if glycopeptide is not None:
            self.glycopeptide = str(glycopeptide)
            self.id = glycopeptide.id
            self.total_mass = glycopeptide.total_mass
        else:
            self.glycopeptide = ''
            self.id = None
            self.total_mass = 0

    def __repr__(self):
        return "Record(%s)" % self.glycopeptide

    def __getstate__(self):
        return (self.glycopeptide, self.id, self.total_mass)

    def __setstate__(self, state):
        self.glycopeptide, self.id, self.total_mass = state

    def __eq__(self, other):
        if other is None:
            return False
        return (self.glycopeptide == other.glycopeptide) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    def convert(self):
        if self.id.structure_type & StructureClassification.target_peptide_decoy_glycan.value:
            struct = DecoyFragmentCachingGlycopeptide(self.glycopeptide)
        else:
            struct = FragmentCachingGlycopeptide(self.glycopeptide)
        struct.id = self.id
        return struct

    @classmethod
    def build(cls, glycopeptides):
        return [cls(p) for p in glycopeptides]

    def copy(self, structure_type=None):
        if structure_type is None:
            structure_type = self.id.structure_type
        inst = self.__class__()
        inst.id = self.id.copy(structure_type)
        inst.glycopeptide = self.glycopeptide
        inst.total_mass = self.total_mass
        return inst

    def to_decoy_glycan(self):
        inst = self.__class__()
        inst.id = self.id.to_decoy_glycan()
        inst.glycopeptide = self.glycopeptide
        inst.total_mass = self.total_mass
        return inst


class Parser(object):
    def __call__(self, record):
        struct = record.convert()
        return struct
