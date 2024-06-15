import os
import operator
import logging
import struct
import ctypes

from typing import Dict, List, Optional, Set

from collections import namedtuple, defaultdict

from glycopeptidepy import PeptideSequence, GlycosylationType
from glycopeptidepy.structure.glycan import GlycanCompositionWithOffsetProxy
from glycopeptidepy.structure.sequence import (
    _n_glycosylation,
    _o_glycosylation,
    _gag_linker_glycosylation,
)

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue
from glypy.utils.enum import EnumValue

from ms_deisotope import isotopic_shift
from ms_deisotope.data_source import ProcessedScan

from glycresoft.chromatogram_tree.mass_shift import MassShift
from glycresoft.serialize.hypothesis.glycan import GlycanTypes

from glycresoft.task import LoggingMixin

from glycresoft.chromatogram_tree import Unmodified

from glycresoft.structure.lru import LRUMapping
from glycresoft.structure import (
    FragmentCachingGlycopeptide,
    DecoyFragmentCachingGlycopeptide,
    PeptideProteinRelation,
)
from glycresoft.structure.structure_loader import (
    CachingStubGlycopeptideStrategy,
    PeptideDatabaseRecord,
)

from glycresoft.tandem.oxonium_ions import (
    OxoniumIndex,
    SignatureIonIndex,
    SignatureIonIndexMatch,
    gscore_scanner,
)

from glycresoft.composition_distribution_model.site_model import GlycoproteomePriorAnnotator

from glycresoft.database import intervals
from glycresoft.database.mass_collection import NeutralMassDatabase, SearchableMassCollection
from glycresoft.database.builder.glycopeptide.common import limiting_combinations

from glycresoft.structure.enums import SpectrumMatchClassification as StructureClassification

from ...workload import WorkloadManager
from ..core_search import GlycanCombinationRecord, IndexedGlycanFilteringPeptideMassEstimator


logger = logging.Logger("glycresoft.dynamic_generation")


debug_mode = bool(os.environ.get("GLYCRESOFTDEBUG"))


_glycopeptide_key_t = namedtuple(
    "glycopeptide_key",
    (
        "start_position",
        "end_position",
        "peptide_id",
        "protein_id",
        "hypothesis_id",
        "glycan_combination_id",
        "structure_type",
        "site_combination_index",
        "glycosylation_type",
    ),
)


class glycopeptide_key_t_base(_glycopeptide_key_t):
    __slot__ = ()

    def copy(self, structure_type=None):
        if structure_type is None:
            structure_type = self.structure_type
        return self._replace(structure_type=structure_type)

    def as_dict(self, stringify=False):
        d = {}
        for i, label in enumerate(self._fields):
            d[label] = str(self[i]) if stringify else self[i]
        return d


MonosaccharideResidues = Set[FrozenMonosaccharideResidue]
GlycanCombinationRecords = List[GlycanCombinationRecord]

# try:
#     from glycresoft._c.structure.structure_loader import glycopeptide_key as glycopeptide_key_t_base
# except ImportError:
#     pass

# The site_combination_index slot is necessary to distinguish alternative arrangements of
# the same combination of glycans on the same amino acid sequence. The placeholder value
# used for unassigned glycosite combination permutations, the maximum value that fits in
# an unsigned 4 byte integer (L signifier in struct spec). This means defines the upper
# limit on the number of distinct combinations that can be uniquely addressed is
# 2 ** 32 - 1 (4294967295).
#
# Note that the glycoform generators all use (circa 2018) :func:`~.limiting_combinations`,
# which only produces the first 100 iterations to avoid producing too many permutations of
# O-glycans.
placeholder_permutation = ctypes.c_uint32(-1).value

TT = StructureClassification.target_peptide_target_glycan


class glycopeptide_key_t(glycopeptide_key_t_base):
    __slots__ = ()

    start_position: int
    end_position: int
    peptide_id: int
    protein_id: int
    hypothesis_id: int
    glycan_combination_id: int
    structure_type: EnumValue
    site_combination_index: int
    glycosylation_type: str

    struct_spec = struct.Struct("!LLLLLLLL")

    def serialize(self):
        return self.struct_spec.pack(*self)

    @classmethod
    def parse(cls, binary):
        return cls(*cls.struct_spec.unpack(binary))

    def to_decoy_glycan(self) -> "glycopeptide_key_t":
        structure_type = self.structure_type
        new_tp = StructureClassification[
            structure_type | StructureClassification.target_peptide_decoy_glycan
        ]
        return self.copy(new_tp)

    def is_decoy(self) -> bool:
        return self.structure_type != 0


CORE_TO_GLYCOSYLATION_TYPE = {
    _n_glycosylation: GlycanTypes.n_glycan,
    _o_glycosylation: GlycanTypes.o_glycan,
    _gag_linker_glycosylation: GlycanTypes.gag_linker,
}


class GlycoformGeneratorBase(LoggingMixin):
    glycan_combinations: NeutralMassDatabase[GlycanCombinationRecord]
    _peptide_cache: LRUMapping
    default_structure_type: StructureClassification
    glycan_prior_model: GlycoproteomePriorAnnotator
    expand_combinatorics: bool

    @classmethod
    def from_hypothesis(cls, session, hypothesis_id, **kwargs):
        """Build a glycan combination index from a :class:`~.GlycanHypothesis`

        Parameters
        ----------
        session: :class:`DatabaseBoundOperation`
            The database connection to use to load glycan combinations.
        hypothesis_id: int
            The id key of the :class:`~.GlycanHypothesis` to build from

        Returns
        -------
        :class:`GlycoformGeneratorBase`
        """
        glycan_combinations = GlycanCombinationRecord.from_hypothesis(session, hypothesis_id)
        return cls(glycan_combinations, **kwargs)

    def __init__(
        self,
        glycan_combinations,
        cache_size=None,
        default_structure_type=TT,
        expand_combinatorics: bool = True,
        *args,
        **kwargs
    ):
        if not isinstance(glycan_combinations, SearchableMassCollection):
            glycan_combinations = NeutralMassDatabase(
                list(glycan_combinations), operator.attrgetter("dehydrated_mass")
            )
        if cache_size is None:
            cache_size = 2**15
        else:
            cache_size = int(cache_size)
        self.glycan_combinations = glycan_combinations
        self._peptide_cache = LRUMapping(cache_size)
        self._cache_hit = 0
        self._cache_miss = 0
        self.default_structure_type = default_structure_type
        self.glycan_prior_model = kwargs.pop("glycan_prior_model", None)
        self.expand_combinatorics = expand_combinatorics
        super(GlycoformGeneratorBase, self).__init__(*args, **kwargs)

    def handle_glycan_combination(
        self,
        peptide_obj: PeptideSequence,
        peptide_record: PeptideDatabaseRecord,
        glycan_combination: GlycanCombinationRecord,
        glycosylation_sites: List[int],
        core_type: GlycosylationType,
    ) -> List["Record"]:
        glycosylation_type = CORE_TO_GLYCOSYLATION_TYPE[core_type]
        key = self._make_key(peptide_record, glycan_combination, glycosylation_type)
        if key in self._peptide_cache:
            self._cache_hit += 1
            return self._peptide_cache[key]
        self._cache_miss += 1
        protein_relation = PeptideProteinRelation(
            key.start_position, key.end_position, key.protein_id, key.hypothesis_id
        )
        glycosylation_sites_unoccupied = set(glycosylation_sites)
        for site in list(glycosylation_sites_unoccupied):
            if peptide_obj[site][1]:
                glycosylation_sites_unoccupied.remove(site)

        site_combinations = list(
            limiting_combinations(
                glycosylation_sites_unoccupied,
                glycan_combination.count,
                100 if self.expand_combinatorics else 1
            )
        )

        result_set = [None for i in site_combinations]
        for i, site_set in enumerate(site_combinations):
            glycoform = peptide_obj.clone(share_cache=False)
            glycoform.id = key._replace(site_combination_index=i)
            glycoform.glycan = GlycanCompositionWithOffsetProxy(glycan_combination.composition)
            for site in site_set:
                glycoform.add_modification(site, core_type.name)
            glycoform.protein_relation = protein_relation
            if self.glycan_prior_model is not None:
                glycoform.glycan_prior = self.glycan_prior_model.score(
                    glycoform, key.structure_type
                )
            result_set[i] = glycoform
        result_set = Record.build(result_set)
        self._peptide_cache[key] = result_set
        return result_set

    def _make_key(
        self,
        peptide_record: PeptideDatabaseRecord,
        glycan_combination: GlycanCombinationRecord,
        glycosylation_type: str,
        structure_type: StructureClassification = None,
    ) -> glycopeptide_key_t:
        if structure_type is None:
            structure_type = self.default_structure_type
        key = glycopeptide_key_t(
            peptide_record.start_position,
            peptide_record.end_position,
            peptide_record.id,
            peptide_record.protein_id,
            peptide_record.hypothesis_id,
            glycan_combination.id,
            structure_type,
            placeholder_permutation,
            glycosylation_type,
        )
        return key

    def handle_n_glycan(self, peptide_obj, peptide_record, glycan_combination):
        if glycan_combination.size < 5:
            return []
        return self.handle_glycan_combination(
            peptide_obj,
            peptide_record,
            glycan_combination,
            peptide_record.n_glycosylation_sites,
            _n_glycosylation,
        )

    def handle_o_glycan(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj,
            peptide_record,
            glycan_combination,
            peptide_record.o_glycosylation_sites,
            _o_glycosylation,
        )

    def handle_gag_linker(self, peptide_obj, peptide_record, glycan_combination):
        return self.handle_glycan_combination(
            peptide_obj,
            peptide_record,
            glycan_combination,
            peptide_record.gagylation_sites,
            _gag_linker_glycosylation,
        )

    def reset_cache(self, **kwargs):
        self._cache_hit = 0
        self._cache_miss = 0
        self._peptide_cache.clear()

    def reset(self, **kwargs):
        self.reset_cache(**kwargs)


class PeptideGlycosylator(GlycoformGeneratorBase):
    peptides: NeutralMassDatabase
    peptide_to_group_id: Dict[int, List[int]]

    def __init__(
        self,
        peptide_records,
        glycan_combinations,
        cache_size=2**16,
        default_structure_type=TT,
        *args,
        **kwargs
    ):
        super(PeptideGlycosylator, self).__init__(
            glycan_combinations,
            default_structure_type=default_structure_type,
            cache_size=cache_size,
            *args,
            **kwargs
        )
        if not isinstance(peptide_records, SearchableMassCollection):
            peptide_records = NeutralMassDatabase(peptide_records)
        self.peptides = peptide_records
        self.peptide_to_group_id = None

    def build_peptide_groups(self):
        peptide_groups = defaultdict(list)
        for peptide in self.peptides:
            peptide_groups[peptide.modified_peptide_sequence].append(peptide.id)

        sequence_to_group_id = {}
        for i, key in enumerate(peptide_groups):
            sequence_to_group_id[key] = i

        peptide_to_group_id = {}
        for peptide in self.peptides:
            peptide_to_group_id[peptide.id] = sequence_to_group_id[
                peptide.modified_peptide_sequence
            ]
        sequence_to_group_id.clear()
        peptide_groups.clear()
        self.peptide_to_group_id = peptide_to_group_id

    def filter_glycan_proposals_by_signature_ions(
        self,
        glycan_combinations: GlycanCombinationRecords,
        signature_ion_match_index: SignatureIonIndexMatch,
    ) -> GlycanCombinationRecords:
        """Remove any glycans for which an expected signature ion is absent (or where the best matching peak is
        less than 1% of the base peak). Requires the signature ion index is initialized.
        """
        result = []
        for glycan in glycan_combinations:
            rec = signature_ion_match_index.record_for(glycan.composition)
            if rec.expected_matches:
                miss = 0
                for _sig, peak in rec.expected_matches.items():
                    if peak is None:
                        miss += 1
                    elif peak.intensity / signature_ion_match_index.base_peak_intensity < 0.01:
                        miss += 1
                if miss:
                    continue
            result.append(glycan)
        return result

    def handle_peptide_mass(
        self,
        peptide_mass: float,
        intact_mass: float,
        error_tolerance: float = 1e-5,
        signature_ion_match_index: SignatureIonIndexMatch = None,
    ) -> List["Record"]:
        peptide_records = self.peptides.search_mass_ppm(peptide_mass, error_tolerance)
        glycan_mass = intact_mass - peptide_mass
        glycan_combinations = self.glycan_combinations.search_mass_ppm(glycan_mass, error_tolerance)
        if signature_ion_match_index:
            glycan_combinations = self.filter_glycan_proposals_by_signature_ions(
                glycan_combinations, signature_ion_match_index
            )
        result_set = []
        for peptide in peptide_records:
            self._combinate(peptide, glycan_combinations, result_set)
        return result_set

    def _combinate(
        self,
        peptide: PeptideDatabaseRecord,
        glycan_combinations: GlycanCombinationRecords,
        result_set: Optional[List["Record"]] = None,
    ) -> List["Record"]:
        if result_set is None:
            result_set = []
        peptide_obj = peptide.convert()
        for glycan_combination in glycan_combinations:
            for tp in glycan_combination.glycan_types:
                tp = GlycosylationType[tp]
                if tp is GlycosylationType.n_linked:
                    result_set.extend(
                        self.handle_n_glycan(peptide_obj.clone(), peptide, glycan_combination)
                    )
                elif tp is GlycosylationType.o_linked:
                    result_set.extend(
                        self.handle_o_glycan(peptide_obj.clone(), peptide, glycan_combination)
                    )
                elif tp is GlycosylationType.glycosaminoglycan:
                    result_set.extend(
                        self.handle_gag_linker(peptide_obj.clone(), peptide, glycan_combination)
                    )
        return result_set

    def generate_crossproduct(self, lower_bound=0, upper_bound=float("inf")):
        minimum_peptide_mass = max(lower_bound - self.glycan_combinations.highest_mass, 0)
        maximum_peptide_mass = max(upper_bound - self.glycan_combinations.lowest_mass, 0)
        peptides = self.peptides.search_between(
            max(minimum_peptide_mass - 10, 0), maximum_peptide_mass
        )
        for peptide in peptides:
            if not peptide.has_glycosylation_sites():
                continue
            glycan_mass_limit = upper_bound - peptide.calculated_mass
            if glycan_mass_limit < 0:
                continue
            minimum_glycan_mass = max(lower_bound - peptide.calculated_mass, 0)
            glycan_combinations = self.glycan_combinations.search_between(
                minimum_glycan_mass, glycan_mass_limit + 10
            )
            for solution in self._combinate(peptide, glycan_combinations):
                total_mass = solution.total_mass
                if total_mass < lower_bound or total_mass > upper_bound:
                    continue
                yield solution

    def reset(self, **kwargs):
        super(PeptideGlycosylator, self).reset(**kwargs)
        self.peptides.reset(**kwargs)


class DynamicGlycopeptideSearchBase(LoggingMixin):
    neutron_shift = isotopic_shift()

    def handle_scan_group(
        self,
        group: List,
        precursor_error_tolerance: float = 1e-5,
        mass_shifts: Optional[List[MassShift]] = None,
    ):
        raise NotImplementedError()

    def reset(self, **kwargs):
        self.peptide_glycosylator.reset(**kwargs)

    def construct_peptide_groups(self, workload: WorkloadManager):
        if self.peptide_glycosylator.peptide_to_group_id is None:
            self.peptide_glycosylator.build_peptide_groups()
        workload.hit_group_map.clear()
        for hit in workload.hit_map.values():
            workload.hit_group_map[
                self.peptide_glycosylator.peptide_to_group_id[hit.id.peptide_id]
            ].add(hit.id)


class PeptideMassFilterBase:
    peptide_glycosylator: PeptideGlycosylator
    product_error_tolerance: float
    glycan_score_threshold: float
    min_fragments: int
    peptide_mass_predictor: IndexedGlycanFilteringPeptideMassEstimator
    peptide_masses_per_scan: int
    probing_range_for_missing_precursors: int
    trust_precursor_fits: bool
    oxonium_ion_index: OxoniumIndex
    signature_ion_index: SignatureIonIndex
    oxonium_ion_threshold: float

    monosaccharides: MonosaccharideResidues

    neutron_shift: float = isotopic_shift()

    def __init__(
        self,
        glycan_compositions: GlycanCombinationRecords,
        product_error_tolerance: float=2e-5,
        glycan_score_threshold: float=0.1,
        min_fragments: int=2,
        peptide_masses_per_scan: int=100,
        probing_range_for_missing_precursors: int=3,
        trust_precursor_fits: bool=True,
        fragment_weight: float=0.56,
        core_weight: float = 1.42,
        oxonium_ion_index: Optional[OxoniumIndex]=None,
        signature_ion_index: Optional[SignatureIonIndex]=None,
        oxonium_ion_threshold: float = 0.05
    ):
        # Intentionally use a larger core_weight here than in the real scoring function to
        # prefer solutions with more core fragments, but not to discard them later?
        if min_fragments is None:
            min_fragments = 2
        self.product_error_tolerance = product_error_tolerance
        self.glycan_score_threshold = glycan_score_threshold
        self.min_fragments = int(min_fragments)
        self.peptide_mass_predictor = IndexedGlycanFilteringPeptideMassEstimator(
            glycan_compositions,
            product_error_tolerance=product_error_tolerance,
            fragment_weight=fragment_weight,
            core_weight=core_weight,
        )
        self.monosaccharides = self._monosaccharides_from_records(glycan_compositions)
        self.peptide_masses_per_scan = peptide_masses_per_scan
        self.probing_range_for_missing_precursors = probing_range_for_missing_precursors
        self.trust_precursor_fits = trust_precursor_fits
        self.oxonium_ion_index = oxonium_ion_index
        self.signature_ion_index = signature_ion_index
        self.oxonium_ion_threshold = oxonium_ion_threshold

    def _monosaccharides_from_records(self, glycan_combinations: GlycanCombinationRecords) -> MonosaccharideResidues:
        residues = set()
        for rec in glycan_combinations:
            residues.update(rec.composition)
        return {FrozenMonosaccharideResidue.from_iupac_lite(str(x)) for x in residues}


class PredictiveGlycopeptideSearch(PeptideMassFilterBase, DynamicGlycopeptideSearchBase):
    peptide_glycosylator: PeptideGlycosylator

    def __init__(
        self,
        peptide_glycosylator,
        product_error_tolerance=2e-5,
        glycan_score_threshold=0.1,
        min_fragments=2,
        peptide_masses_per_scan=100,
        probing_range_for_missing_precursors=3,
        trust_precursor_fits=True,
        fragment_weight=0.56,
        core_weight=1.42,
        oxonium_ion_index=None,
        signature_ion_index=None,
        oxonium_ion_threshold: float=0.05
    ):
        self.peptide_glycosylator = peptide_glycosylator
        super().__init__(
            self.peptide_glycosylator.glycan_combinations,
            product_error_tolerance=product_error_tolerance,
            glycan_score_threshold=glycan_score_threshold,
            min_fragments=min_fragments,
            peptide_masses_per_scan=peptide_masses_per_scan,
            probing_range_for_missing_precursors=probing_range_for_missing_precursors,
            trust_precursor_fits=trust_precursor_fits,
            fragment_weight=fragment_weight,
            core_weight=core_weight,
            oxonium_ion_index=oxonium_ion_index,
            signature_ion_index=signature_ion_index,
            oxonium_ion_threshold=oxonium_ion_threshold)

    def handle_scan_group(
        self,
        group: List[ProcessedScan],
        precursor_error_tolerance: float = 1e-5,
        mass_shifts: Optional[List[MassShift]] = None,
        workload: Optional[WorkloadManager] = None,
    ):
        if mass_shifts is None or not mass_shifts:
            mass_shifts = [Unmodified]
        if workload is None:
            workload = WorkloadManager()

        glycan_score_threshold = self.glycan_score_threshold
        min_fragments = self.min_fragments
        estimate_peptide_mass = self.peptide_mass_predictor.estimate_peptide_mass
        handle_peptide_mass = self.peptide_glycosylator.handle_peptide_mass
        peptide_masses_per_scan = self.peptide_masses_per_scan
        for scan in group:
            if gscore_scanner(scan) < self.oxonium_ion_threshold:
                continue
            workload.add_scan(scan)
            signature_ion_matches: SignatureIonIndexMatch = None

            if self.oxonium_ion_index:
                scan.annotations["oxonium_index_match"] = self.oxonium_ion_index.match(
                    scan.deconvoluted_peak_set, self.product_error_tolerance
                )
            if self.signature_ion_index:
                signature_ion_matches = scan.annotations[
                    "signature_index_match"
                ] = self.signature_ion_index.match(
                    scan.deconvoluted_peak_set, self.product_error_tolerance
                )
            n_glycopeptides = 0
            n_peptide_masses = 0
            if not scan.precursor_information.defaulted and self.trust_precursor_fits:
                probe = 0
            else:
                probe = self.probing_range_for_missing_precursors
            precursor_mass = scan.precursor_information.neutral_mass
            for i in range(probe + 1):
                neutron_shift = self.neutron_shift * i
                for mass_shift in mass_shifts:
                    seen = set()
                    mass_shift_name = mass_shift.name
                    mass_shift_mass = mass_shift.mass
                    intact_mass = precursor_mass - mass_shift_mass - neutron_shift
                    for peptide_mass_pred in estimate_peptide_mass(
                        scan,
                        topn=peptide_masses_per_scan,
                        mass_shift=mass_shift,
                        threshold=glycan_score_threshold,
                        min_fragments=min_fragments,
                        simplify=False,
                        query_mass=precursor_mass - neutron_shift,
                    ):
                        peptide_mass = peptide_mass_pred.peptide_mass
                        n_peptide_masses += 1
                        for candidate in handle_peptide_mass(
                            peptide_mass,
                            intact_mass,
                            self.product_error_tolerance,
                            signature_ion_match_index=signature_ion_matches,
                        ):
                            n_glycopeptides += 1
                            key = (candidate.id, mass_shift_name)
                            mass_threshold_passed = (
                                abs(intact_mass - candidate.total_mass) / intact_mass
                            ) <= precursor_error_tolerance
                            # What if it could be an N- or O-glycopeptide? Does this block the
                            # same glycan from being treated as both?
                            if key in seen:
                                continue
                            seen.add(key)
                            if mass_threshold_passed:
                                workload.add_scan_hit(scan, candidate, mass_shift_name)
        return workload


DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT = 1e5


class IterativeGlycopeptideSearch(DynamicGlycopeptideSearchBase):
    total_mass_getter = operator.attrgetter("total_mass")

    def __init__(
        self,
        peptide_glycosylator,
        product_error_tolerance=2e-5,
        cache_size=2,
        threshold_cache_total_count=DEFAULT_THRESHOLD_CACHE_TOTAL_COUNT,
    ):
        self.peptide_glycosylator = peptide_glycosylator
        self.product_error_tolerance = product_error_tolerance
        self.interval_cache = intervals.LRUIntervalSet(max_size=cache_size)
        self.threshold_cache_total_count = threshold_cache_total_count

    def _make_new_glycopeptides_for_interval(self, lower_bound, upper_bound):
        generator = self.peptide_glycosylator.generate_crossproduct(lower_bound, upper_bound)
        generator = NeutralMassDatabase(list(generator), self.total_mass_getter)
        return intervals.ConcatenateMassIntervalNode(generator)

    def _upkeep_memory_intervals(self, lower_bound=0, upper_bound=float("inf")):
        """Perform routine maintainence of the interval cache, ensuring its size does not
        exceed the upper limit
        """
        n = len(self.interval_cache)
        if n > 1:
            while (
                len(self.interval_cache) > 1
                and self.interval_cache.total_count > self.threshold_cache_total_count
            ):
                logger.info(
                    "Upkeep Memory Intervals %d %d",
                    self.interval_cache.total_count,
                    len(self.interval_cache),
                )
                self.interval_cache.remove_lru_interval()
        n = len(self.interval_cache)
        if n == 1 and self.interval_cache.total_count > self.threshold_cache_total_count:
            segment = self.interval_cache[0]
            segment.constrain(lower_bound, upper_bound)

    def get_glycopeptides(self, lower_bound, upper_bound):
        self._upkeep_memory_intervals(lower_bound, upper_bound)
        q = intervals.QueryIntervalBase((lower_bound + upper_bound) / 2.0, lower_bound, upper_bound)
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
                logger.debug(
                    "Query interval partially overlapped, creating disjoint interval %r", q2
                )
                match = self.interval_cache.extend_interval(
                    match, self._make_new_glycopeptides_for_interval(q2.start, q2.end)
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

    def handle_scan_group(
        self, group, precursor_error_tolerance=1e-5, mass_shifts=None, workload=None
    ):
        if mass_shifts is None or not mass_shifts:
            mass_shifts = [Unmodified]
        if workload is None:
            workload = WorkloadManager()

        group = NeutralMassDatabase(group, lambda x: x.precursor_information.neutral_mass)
        lo_mass = group.lowest_mass
        hi_mass = group.highest_mass
        shifts = [m.mass for m in mass_shifts]

        lo_shift = min(shifts) - 0.5
        hi_shift = max(shifts) + 0.5

        lower_bound = lo_mass + lo_shift
        lower_bound = lower_bound - (lower_bound * precursor_error_tolerance)

        upper_bound = hi_mass + hi_shift
        upper_bound = upper_bound + (upper_bound * precursor_error_tolerance)

        id_to_scan = {}
        for scan in group:
            id_to_scan[scan.id] = scan
        logger.info("\nQuerying between %f and %f", lower_bound, upper_bound)
        candidates = self.get_glycopeptides(lower_bound, upper_bound)
        logger.info(
            "Interval from %f to %f contained %d candidates",
            lower_bound,
            upper_bound,
            len(candidates),
        )
        for scan in group:
            for mass_shift in mass_shifts:
                intact_mass = scan.precursor_information.neutral_mass - mass_shift.mass
                for candidate in candidates.search_mass_ppm(intact_mass, precursor_error_tolerance):
                    workload.add_scan_hit(scan, candidate, mass_shift.name)
        return workload


class CompoundGlycopeptideSearch(object):
    def __init__(self, glycopeptide_searchers=None):
        if glycopeptide_searchers is None:
            glycopeptide_searchers = []
        self.glycopeptide_searchers = list(glycopeptide_searchers)

    def add(self, glycopeptide_searcher):
        self.glycopeptide_searchers.append(glycopeptide_searcher)

    def handle_scan_group(
        self, group, precursor_error_tolerance=1e-5, mass_shifts=None, workload=None
    ):
        if mass_shifts is None:
            mass_shifts = [Unmodified]
        if workload is None:
            workload = WorkloadManager()
        for searcher in self.glycopeptide_searchers:
            searcher.handle_scan_group(
                group, precursor_error_tolerance, mass_shifts, workload=workload
            )
        return workload


class Record(object):
    __slots__ = ("glycopeptide", "id", "total_mass", "glycan_prior")

    glycopeptide: str
    id: glycopeptide_key_t
    total_mass: float
    glycan_prior: float

    def __init__(self, glycopeptide: FragmentCachingGlycopeptide = None):
        if glycopeptide is not None:
            self.glycopeptide = str(glycopeptide)
            self.id = glycopeptide.id
            self.total_mass = glycopeptide.total_mass
            self.glycan_prior = glycopeptide.glycan_prior
        else:
            self.glycopeptide = ""
            self.id = None
            self.total_mass = 0
            self.glycan_prior = 0.0

    def __repr__(self):
        return "Record(%s)" % self.glycopeptide

    def __getstate__(self):
        return (self.glycopeptide, self.id, self.total_mass, self.glycan_prior)

    def __setstate__(self, state):
        self.glycopeptide, self.id, self.total_mass, self.glycan_prior = state

    def __eq__(self, other):
        if other is None:
            return False
        return (self.glycopeptide == other.glycopeptide) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    def convert(self) -> FragmentCachingGlycopeptide:
        if self.id.structure_type.value & StructureClassification.target_peptide_decoy_glycan.value:
            structure = SharedCacheAwareDecoyFragmentCachingGlycopeptide(self.glycopeptide)
        else:
            structure = FragmentCachingGlycopeptide(self.glycopeptide)
        structure.id = self.id
        structure.glycan_id = self.id.glycan_combination_id
        structure.glycan_prior = self.glycan_prior
        return structure

    def serialize(self):
        id_bytes = self.id.serialize()
        mass_bytes = struct.pack("!d", self.total_mass)
        seq = (self.glycopeptide).encode("utf8")
        encoded = id_bytes + mass_bytes + seq
        return encoded

    @classmethod
    def parse(cls, bytestring):
        offset = glycopeptide_key_t.struct_spec.size
        rec_id = glycopeptide_key_t.parse(bytestring[:offset])
        mass = struct.unpack("!d", bytestring[offset : offset + 8])
        seq = bytestring[offset + 8 :].decode("utf8")
        inst = cls()
        inst.id = rec_id
        inst.total_mass = mass
        inst.glycopeptide = seq
        return inst

    @classmethod
    def build(cls, glycopeptides: List[FragmentCachingGlycopeptide]) -> List["Record"]:
        return [cls(p) for p in glycopeptides]

    def copy(self, structure_type=None) -> "Record":
        if structure_type is None:
            structure_type = self.id.structure_type
        inst = self.__class__()
        inst.id = self.id.copy(structure_type)
        inst.glycopeptide = self.glycopeptide
        inst.total_mass = self.total_mass
        return inst

    def to_decoy_glycan(self) -> "Record":
        inst = self.__class__()
        inst.id = self.id.to_decoy_glycan()
        inst.glycopeptide = self.glycopeptide
        inst.total_mass = self.total_mass
        return inst


class SharedCacheAwareDecoyFragmentCachingGlycopeptide(DecoyFragmentCachingGlycopeptide):
    def stub_fragments(self, *args, **kwargs):
        kwargs.setdefault("strategy", CachingStubGlycopeptideStrategy)
        key = self.fragment_caches.stub_fragment_key(self, args, kwargs)
        if key in self.fragment_caches:
            return self.fragment_caches[key]
        target_key = self.fragment_caches._make_target_key(key)
        if target_key in self.fragment_caches:
            result = self.fragment_caches[target_key]
            do_clone = True
        else:
            do_clone = False
            result = list(
                # Directly call the superclass method of FragmentCachingGlycopeptide as we
                # do not need to go through a preliminary round of cache key construction and
                # querying.
                super(
                    FragmentCachingGlycopeptide, self
                ).stub_fragments(  # pylint: disable=bad-super-call
                    *args, **kwargs
                )
            )
        result = self._permute_stub_masses(result, kwargs, do_clone=do_clone, min_shift_size=1)
        self.fragment_caches[key] = result
        return result


class Parser(object):
    def __init__(self, max_size=2**12, **kwargs):
        self.cache = LRUMapping(max_size)

    def __call__(self, record):
        key = record.id
        if key in self.cache:
            return self.cache[key]
        else:
            struct = record.convert()
            self.cache[key] = struct
            return struct


class IdKeyMaker(object):
    """Build an id-key structure for a glycopeptide with a new
    glycan composition given a glycopeptide of the same peptide
    backbone.
    """

    def __init__(self, valid_glycans):
        self.valid_glycans = valid_glycans
        self.lookup_map = {g: g.id for g in self.valid_glycans}

    def make_id_controlled_structures(self, structure, references):
        glycan_id = self.lookup_map[structure.glycan_composition]
        result = []
        for reference in references:
            ref_key = reference.id
            alt_key = glycopeptide_key_t(
                ref_key.start_position,
                ref_key.end_position,
                ref_key.peptide_id,
                ref_key.protein_id,
                ref_key.hypothesis_id,
                glycan_id,
                ref_key.structure_type,
                ref_key.site_combination_index,
                # TODO: Does this needs to be generic over all glycosylation types of
                # glycan_id if it could span multiple, or will we assume that all alternatives
                # will be in references?
                ref_key.glycosylation_type,
            )
            alt = structure.clone()
            alt.id = alt_key
            alt.protein_relation = reference.protein_relation
            result.append(alt)
        return result

    def __call__(self, structure, references):
        return self.make_id_controlled_structures(structure, references)
