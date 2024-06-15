# -*- coding: utf-8 -*-

import logging
import warnings

from collections import namedtuple, defaultdict

from typing import Any, DefaultDict, Dict, Generic, Optional, List, Set, Tuple, TypeVar, Union, Sequence

import numpy as np

from ms_deisotope.data_source import ProcessedScan
from ms_deisotope.peak_set import DeconvolutedPeak

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, HashableGlycanComposition, GlycanComposition
from glypy.utils.enum import EnumValue

from glycopeptidepy.structure.fragment import StubFragment
from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy, _AccumulatorBag
from glycopeptidepy.structure.fragmentation_strategy.glycan import GlycanCompositionFragment


from glycresoft.chromatogram_tree.mass_shift import MassShift
from glycresoft.serialize import GlycanCombination, GlycanTypes
from glycresoft.database.disk_backed_database import PPMQueryInterval
from glycresoft.chromatogram_tree import Unmodified
from glycresoft.structure.denovo import PathSet, PathFinder, Path

logger = logging.getLogger("glycresoft.core_search")


hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
xylose = FrozenMonosaccharideResidue.from_iupac_lite("Xyl")
fucose = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")
dhex = FrozenMonosaccharideResidue.from_iupac_lite("dHex")
neuac = FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")
neugc = FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")


def approximate_internal_size_of_glycan(glycan_composition):
    terminal_groups = glycan_composition._getitem_fast(neuac) +\
        glycan_composition._getitem_fast(neugc)
    side_groups = glycan_composition._getitem_fast(fucose) + glycan_composition._getitem_fast(dhex)
    n = sum(glycan_composition.values())
    n -= terminal_groups
    if side_groups > 1:
        n -= 1
    return n


def glycan_side_group_count(glycan_composition):
    side_groups = glycan_composition._getitem_fast(
        fucose) + glycan_composition._getitem_fast(dhex)
    return side_groups


def isclose(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= atol + rtol * abs(b)


default_components = (hexnac, hexose, xylose, fucose,)


class CoreMotifFinder(PathFinder):
    minimum_peptide_mass: float

    def __init__(self, components=None, product_error_tolerance=1e-5, minimum_peptide_mass=350.):
        if components is None:
            components = default_components
        super().__init__(components, product_error_tolerance, False)
        self.minimum_peptide_mass = minimum_peptide_mass

    def find_n_linked_core(self, groups: DefaultDict[Any, List], min_size=1):
        sequence = [hexnac, hexnac, hexose, hexose, hexose]
        expected_n = len(sequence)
        terminals = dict()

        for label, paths in groups.items():
            label_i = 0
            expected_i = 0
            path_n = len(label)
            while label_i < path_n and expected_i < expected_n:
                edge = label[label_i]
                label_i += 1
                expected = sequence[expected_i]
                if expected == edge:
                    expected_i += 1
                elif edge == fucose or edge == dhex:
                    continue
                else:
                    break
            if expected_i >= min_size:
                for path in paths:
                    last_path = terminals.get(path[0].start)
                    if last_path is None:
                        terminals[path[0].start] = path
                    else:
                        terminals[path[0].start] = max((path, last_path), key=lambda x: x.total_signal)
        return PathSet(terminals.values())

    def find_o_linked_core(self, groups: DefaultDict[Any, List], min_size=1):
        sequence = [(hexnac, hexose), (hexnac, hexose, fucose,), (hexnac, hexose, fucose,)]
        expected_n = len(sequence)
        terminals = dict()

        for label, paths in groups.items():
            label_i = 0
            expected_i = 0
            path_n = len(label)
            while label_i < path_n and expected_i < expected_n:
                edge = label[label_i]
                label_i += 1
                expected = sequence[expected_i]
                if edge in expected:
                    expected_i += 1
                else:
                    break
            if expected_i >= min_size:
                for path in paths:
                    last_path = terminals.get(path[0].start)
                    if last_path is None:
                        terminals[path[0].start] = path
                    else:
                        terminals[path[0].start] = max((path, last_path), key=lambda x: x.total_signal)
        return PathSet(terminals.values())

    def find_gag_linker_core(self, groups: DefaultDict[Any, List], min_size=1):
        sequence = [xylose, hexose, hexose, ]
        expected_n = len(sequence)
        terminals = dict()

        for label, paths in groups.items():
            label_i = 0
            expected_i = 0
            path_n = len(label)
            while label_i < path_n and expected_i < expected_n:
                edge = label[label_i]
                label_i += 1
                expected = sequence[expected_i]
                if expected == edge:
                    expected_i += 1
                elif edge == fucose:
                    continue
                else:
                    break
            if expected_i >= min_size:
                for path in paths:
                    last_path = terminals.get(path[0].start)
                    if last_path is None:
                        terminals[path[0].start] = path
                    else:
                        terminals[path[0].start] = max((path, last_path), key=lambda x: x.total_signal)
        return PathSet(terminals.values())

    def estimate_peptide_mass(self, scan: ProcessedScan, topn: int=100, mass_shift: MassShift=Unmodified,
                              query_mass: Optional[float] = None,
                              simplify: bool = True) -> Union[List[float], List[Tuple[float, List[Path]]]]:
        graph = self.build_graph(scan, mass_shift=mass_shift)
        paths = self.paths_from_graph(graph)
        groups = self.aggregate_paths(paths)

        n_linked_paths = self.find_n_linked_core(groups)
        o_linked_paths = self.find_o_linked_core(groups)
        gag_linker_paths = self.find_gag_linker_core(groups)
        peptide_masses = []

        has_tandem_shift = abs(mass_shift.tandem_mass) > 0

        # TODO: split the different motif masses up according to core type efficiently
        # but for now just lump them all together
        for path in n_linked_paths:
            if path.start_mass < self.minimum_peptide_mass:
                continue
            peptide_masses.append((path.start_mass, path))
            if has_tandem_shift:
                peptide_masses.append((path.start_mass - mass_shift.tandem_mass, path))
        for path in o_linked_paths:
            if path.start_mass < self.minimum_peptide_mass:
                continue
            peptide_masses.append((path.start_mass, path))
            if has_tandem_shift:
                peptide_masses.append((path.start_mass - mass_shift.tandem_mass, path))
        for path in gag_linker_paths:
            if path.start_mass < self.minimum_peptide_mass:
                continue
            peptide_masses.append((path.start_mass, path))
            if has_tandem_shift:
                peptide_masses.append((path.start_mass - mass_shift.tandem_mass, path))
        peptide_masses.sort(key=lambda x: x[0])
        result = []
        paths_for = []
        last = 0
        for m, path in peptide_masses:
            if abs(last - m) < 1e-3:
                if not simplify:
                    paths_for.append(path)
                continue
            if simplify:
                result.append(m)
            else:
                paths_for = [path]
                result.append((m, paths_for))
            last = m
        return result[:topn]

    def build_peptide_filter(self, scan, error_tolerance=1e-5, mass_shift=Unmodified, query_mass=None):
        peptide_masses = self.estimate_peptide_mass(
            scan, mass_shift=mass_shift, query_mass=query_mass, simplify=True)

        out = []
        if len(peptide_masses) == 0:
            return IntervalFilter([])
        last = PPMQueryInterval(peptide_masses[0], error_tolerance)
        for point in peptide_masses[1:]:
            interval = PPMQueryInterval(point, error_tolerance)
            if interval.overlaps(last):
                last.extend(interval)
            else:
                out.append(last)
                last = interval
        out.append(last)
        return IntervalFilter(out)


class CoarseStubGlycopeptideFragment(object):
    __slots__ = ['key', 'is_core', 'mass']

    def __init__(self, key, mass, is_core):
        self.key = key
        self.mass = mass
        self.is_core = is_core

    def __eq__(self, other):
        try:
            return self.key == other.key and self.is_core == other.is_core
        except AttributeError:
            return self.key == other

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(int(self.mass))

    def __lt__(self, other):
        return self.mass < other.mass

    def __gt__(self, other):
        return self.mass > other.mass

    def __reduce__(self):
        return self.__class__, (self.key, self.mass, self.is_core)

    def __repr__(self):
        return "%s(%s, %f, %r)" % (
            self.__class__.__name__,
            self.key, self.mass, self.is_core
        )


class GlycanCombinationRecordBase(object):
    __slots__ = ['id', 'dehydrated_mass', 'composition', 'count', 'glycan_types',
                 'size', "_fragment_cache", "internal_size_approximation", "_hash",
                 'fragment_set_properties']

    id: int
    dehydrated_mass: float
    composition: GlycanComposition
    count: int
    glycan_types: List[EnumValue]
    size: int
    _fragment_cache: Dict[EnumValue, List]
    internal_size_approximation: int
    _hash: int
    fragment_set_properties: dict

    def is_n_glycan(self) -> bool:
        return GlycanTypes.n_glycan in self.glycan_types

    def is_o_glycan(self) -> bool:
        return GlycanTypes.o_glycan in self.glycan_types

    def is_gag_linker(self) -> bool:
        return GlycanTypes.gag_linker in self.glycan_types

    def get_n_glycan_fragments(self) -> List[CoarseStubGlycopeptideFragment]:
        if GlycanTypes.n_glycan not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.n_glycan_composition_fragments(
                self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                if shift["key"]['HexNAc'] <= 2 and shift["key"]["Hex"] <= 3:
                    is_core = True
                else:
                    is_core = False
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], is_core))
            self._fragment_cache[GlycanTypes.n_glycan] = sorted(
                set(fragment_structs))
            return self._fragment_cache[GlycanTypes.n_glycan]
        else:
            return self._fragment_cache[GlycanTypes.n_glycan]

    def get_o_glycan_fragments(self) -> List[CoarseStubGlycopeptideFragment]:
        if GlycanTypes.o_glycan not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.o_glycan_composition_fragments(
                self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                shift['key'] = _AccumulatorBag(shift['key'])
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            self._fragment_cache[GlycanTypes.o_glycan] = sorted(
                set(fragment_structs))
            return self._fragment_cache[GlycanTypes.o_glycan]
        else:
            return self._fragment_cache[GlycanTypes.o_glycan]

    def get_gag_linker_glycan_fragments(self) -> List[CoarseStubGlycopeptideFragment]:
        if GlycanTypes.gag_linker not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.gag_linker_composition_fragments(
                self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                shift['key'] = _AccumulatorBag(shift['key'])
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            self._fragment_cache[GlycanTypes.gag_linker] = sorted(
                set(fragment_structs))
            return self._fragment_cache[GlycanTypes.gag_linker]
        else:
            return self._fragment_cache[GlycanTypes.gag_linker]

    def clear(self):
        self._fragment_cache.clear()

try:
    _GlycanCombinationRecordBase = GlycanCombinationRecordBase
    from glycresoft._c.tandem.core_search import GlycanCombinationRecordBase
except ImportError as err:
    print(err)
    pass


class GlycanCombinationRecord(GlycanCombinationRecordBase):
    """Represent a glycan combination compactly in memory

    Attributes
    ----------
    composition : :class:`~.HashableGlycanComposition`
        The glycan combination's composition in monosaccharide units
    count : int
        The number of distinct glycans this combination contains
    dehydrated_mass : float
        The total mass shift applied to a peptide when this combination is attached
        to it
    glycan_types : list
        The types of glycans combined to make this entity
    """

    __slots__ = ()

    @classmethod
    def from_combination(cls, combination):
        inst = cls(
            id=combination.id,
            dehydrated_mass=combination.dehydrated_mass(),
            composition=combination.convert(),
            count=combination.count,
            glycan_types=tuple(set([
                c.name for component_classes in combination.component_classes
                for c in component_classes])),
        )
        return inst

    @classmethod
    def from_hypothesis(cls, session, hypothesis_id):
        query = session.query(GlycanCombination).filter(
            GlycanCombination.hypothesis_id == hypothesis_id).group_by(
                GlycanCombination.composition, GlycanCombination.count).order_by(
                    GlycanCombination.dehydrated_mass()) # pylint: disable=no-value-for-parameter
        candidates = query.all()
        out = []
        for candidate in candidates:
            out.append(cls.from_combination(candidate))
        return out

    def _to_dict(self):
        return {
            "id": self.id,
            "dehydrated_mass": self.dehydrated_mass,
            "composition": str(self.composition),
            "count": self.count,
            "glycan_types": list(map(str, self.glycan_types)),
        }

    @classmethod
    def _from_dict(cls, d):
        d['composition'] = HashableGlycanComposition.parse(d['composition'])
        d['glycan_types'] = [GlycanTypes[t] for t in d['glycan_types']]
        return cls(**d)

    def __init__(self, id, dehydrated_mass, composition, count, glycan_types):
        self.id = id
        self.dehydrated_mass = dehydrated_mass
        self.composition = composition
        self.size = sum(composition.values())
        self.internal_size_approximation = self._approximate_total_size()
        self.side_group_count = glycan_side_group_count(self.composition)
        self.count = count
        self.glycan_types = list(glycan_types)
        self._fragment_cache = dict()
        self._hash = hash(self.composition)
        self.fragment_set_properties = dict()

    def __eq__(self, other):
        return (self.composition == other.composition) and (self.count == other.count) and (
            self.glycan_types == other.glycan_types)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return self._hash

    def _approximate_total_size(self):
        return approximate_internal_size_of_glycan(self.composition)

    def __reduce__(self):
        return (
            GlycanCombinationRecord,
            (self.id, self.dehydrated_mass, self.composition, self.count, self.glycan_types),
            self.__getstate__()
        )

    def __getstate__(self):
        return {
            "fragment_set_properties": self.fragment_set_properties
        }

    def __setstate__(self, state):
        self.fragment_set_properties = state['fragment_set_properties']

    def __repr__(self):
        return "GlycanCombinationRecord(%s, %d)" % (self.composition, self.count)


class CoarseStubGlycopeptideMatch(object):
    def __init__(self, key, mass, shift_mass, peaks_matched):
        self.key = key
        self.mass = mass
        self.shift_mass = shift_mass
        self.peaks_matched = peaks_matched

    def __reduce__(self):
        return self.__class__, (self.key, self.mass, self.shift_mass, self.peaks_matched)

    def __repr__(self):
        return "%s(%s, %f, %f, %r)" % (
            self.__class__.__name__,
            self.key, self.mass, self.shift_mass, self.peaks_matched
        )


class CoarseGlycanMatch(object):
    def __init__(self, matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical):
        self.fragment_matches = list(matched_fragments)
        self.n_matched = n_matched
        self.n_theoretical = n_theoretical
        self.core_matched = core_matched
        self.core_theoretical = core_theoretical

    def __iter__(self):
        yield self.matched_fragments
        yield self.n_matched
        yield self.n_theoretical
        yield self.core_matched
        yield self.core_theoretical

    def estimate_peptide_mass(self):
        weighted_mass_acc = 0.0
        weight_acc = 0.0

        for fmatch in self.fragment_matches:
            fmass = fmatch.shift_mass
            for peak in fmatch.peaks_matched:
                weighted_mass_acc += (peak.neutral_mass - fmass) * peak.intensity
                weight_acc += peak.intensity
        if weight_acc == 0:
            return -1
        return weighted_mass_acc / weight_acc

    def __repr__(self):
        template = (
            "{self.__class__.__name__}({self.n_matched}, {self.n_theoretical}, "
            "{self.core_matched}, {self.core_theoretical})")
        return template.format(self=self)


class GlycanCoarseScorerBase(object):
    product_error_tolerance: float
    fragment_weight: float
    core_weight: float

    def __init__(self, product_error_tolerance=1e-5, fragment_weight=0.56, core_weight=0.42):
        self.product_error_tolerance = product_error_tolerance
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight

    def _match_fragments(self, scan, peptide_mass, shifts, mass_shift_tandem_mass=0.0):
        fragment_matches = []
        core_matched = 0.0
        core_theoretical = 0.0
        has_tandem_shift = abs(mass_shift_tandem_mass) > 0
        for shift in shifts:
            if shift.is_core:
                is_core = True
                core_theoretical += 1
            else:
                is_core = False
            target_mass = shift.mass + peptide_mass
            hits = scan.deconvoluted_peak_set.all_peaks_for(target_mass, self.product_error_tolerance)
            if hits:
                if is_core:
                    core_matched += 1
                fragment_matches.append((shift.key, target_mass, hits))
            if has_tandem_shift:
                shifted_mass = target_mass + mass_shift_tandem_mass
                hits = scan.deconvoluted_peak_set.all_peaks_for(
                    shifted_mass, self.product_error_tolerance)
                if hits:
                    if is_core:
                        core_matched += 1
                    fragment_matches.append(
                        CoarseStubGlycopeptideMatch(
                            shift.key, shifted_mass, shift.mass + mass_shift_tandem_mass, hits))

        return CoarseGlycanMatch(
            fragment_matches, float(len(fragment_matches)), float(len(shifts)), core_matched, core_theoretical)

    # consider adding the internal size approximation to this method and it's Cython implementation.
    def _calculate_score(self, glycan_match):
        ratio_fragments = (glycan_match.n_matched / glycan_match.n_theoretical)
        ratio_core = glycan_match.core_matched / glycan_match.core_theoretical
        coverage = (ratio_fragments ** self.fragment_weight) * (ratio_core ** self.core_weight)
        score = 0
        for fmatch in glycan_match.fragment_matches:
            mass = fmatch.mass
            for peak in fmatch.matches:
                score += np.log(peak.intensity) * (1 - (np.abs(peak.neutral_mass - mass) / mass) ** 4) * coverage
        return score

    def _n_glycan_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_n_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    def _o_glycan_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_o_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    def _gag_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_gag_linker_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

try:
    _GlycanCoarseScorerBase = GlycanCoarseScorerBase
    from glycresoft._c.tandem.core_search import GlycanCoarseScorerBase
except ImportError:
    pass


_GlycanMatchResult = namedtuple('GlycanMatchResult', (
    'peptide_mass', 'score', 'match', 'glycan_size', 'glycan_types',
    'recalibrated_peptide_mass'))


class GlycanMatchResult(_GlycanMatchResult):
    __slots__ = ()

    @property
    def fragment_match_count(self):
        match = self.match
        if match is None:
            return 0
        return match.n_matched


def group_by_score(matches, threshold=1e-2):
    matches = sorted(matches, key=lambda x: x.score, reverse=True)
    groups = []
    if len(matches) == 0:
        return groups
    current_group = [matches[0]]
    last_match = matches[0]
    for match in matches[1:]:
        delta = abs(match.score - last_match.score)
        if delta > threshold:
            groups.append(current_group)
            current_group = [match]
        else:
            current_group.append(match)
        last_match = match
    groups.append(current_group)
    return groups


def flatten(groups):
    return [b for a in groups for b in a]


SolutionType = TypeVar("SolutionType")


class GlycanFilteringPeptideMassEstimatorBase(Generic[SolutionType]):
    use_denovo_motif: bool
    motif_finder: CoreMotifFinder
    glycan_combination_db: List[GlycanCombinationRecord]
    minimum_peptide_mass: float
    use_recalibrated_peptide_mass: float

    def __init__(self, glycan_combination_db, product_error_tolerance=1e-5,
                 fragment_weight=0.56, core_weight=0.42, minimum_peptide_mass=500.0,
                 use_denovo_motif=False, components=None,
                 use_recalibrated_peptide_mass=False):
        if not isinstance(glycan_combination_db[0], GlycanCombinationRecord):
            glycan_combination_db = [GlycanCombinationRecord.from_combination(gc)
                                     for gc in glycan_combination_db]
        self.use_denovo_motif = use_denovo_motif
        self.motif_finder = CoreMotifFinder(
            components, product_error_tolerance)
        self.glycan_combination_db = sorted(
            glycan_combination_db, key=lambda x: (x.dehydrated_mass, x.id))
        self.minimum_peptide_mass = minimum_peptide_mass
        self.use_recalibrated_peptide_mass = use_recalibrated_peptide_mass
        super(GlycanFilteringPeptideMassEstimatorBase, self).__init__(
            product_error_tolerance, fragment_weight, core_weight)

    def match(self, scan: ProcessedScan, mass_shift=Unmodified, query_mass=None) -> List[SolutionType]:
        raise NotImplementedError()

    def estimate_peptide_mass(self, scan: ProcessedScan,
                              topn: int=150,
                              threshold: float=-1,
                              min_fragments: int=0,
                              mass_shift: MassShift=Unmodified,
                              simplify: bool=True,
                              query_mass: Optional[float] = None) -> Union[List[float], List[SolutionType]]:
        '''Given an scan, estimate the possible peptide masses using the connected glycan database and
        mass differences from the precursor mass.

        Parameters
        ----------
        scan : ProcessedScan
            The deconvoluted scan to search
        topn : int, optional
            The number of solutions to return, sorted by quality descending
        threshold : float, optional
            The minimum match score to allow a returned solution to have.
        min_fragments : int, optional
            The minimum number of matched fragments to require a solution to have, independent
            of score.
        mass_shift : MassShift, optional
            The mass shift to apply to
        '''
        out = self.match(scan, mass_shift=mass_shift, query_mass=query_mass)
        out = [x for x in out if x.score >
               threshold and x.fragment_match_count >= min_fragments]
        groups = group_by_score(out)
        out = flatten(groups[:topn])
        if simplify:
            return [x.peptide_mass for x in out]
        return out

    def glycan_for_peptide_mass(self, scan: ProcessedScan, peptide_mass: float) -> List[GlycanCombinationRecord]:
        matches = []
        try:
            glycan_mass = scan.precursor_information.neutral_mass - peptide_mass
        except AttributeError:
            glycan_mass = scan - peptide_mass
        for glycan_record in self.glycan_combination_db:
            if abs(glycan_record.dehydrated_mass - glycan_mass) / glycan_mass < self.product_error_tolerance:
                matches.append(glycan_record)
            elif glycan_mass > glycan_record.dehydrated_mass:
                break
        return matches

    def build_peptide_filter(self, scan: ProcessedScan,
                             error_tolerance: Optional[float] = None,
                             mass_shift: MassShift = Unmodified,
                             query_mass: Optional[float] = None) -> 'IntervalFilter':
        if error_tolerance is None:
            error_tolerance = self.product_error_tolerance
        peptide_masses = self.estimate_peptide_mass(
            scan, mass_shift=mass_shift, query_mass=query_mass)
        peptide_masses = [PPMQueryInterval(
            p, error_tolerance) for p in peptide_masses]
        if self.use_denovo_motif:
            path_masses = self.motif_finder.build_peptide_filter(
                scan, error_tolerance, mass_shift=mass_shift)
            peptide_masses.extend(path_masses)
        peptide_masses.sort(key=lambda x: x.center)

        if len(peptide_masses) == 0:
            return IntervalFilter([])
        out = IntervalFilter(peptide_masses)
        out.compress()
        return out


class GlycanFilteringPeptideMassEstimator(
        GlycanFilteringPeptideMassEstimatorBase[GlycanMatchResult], GlycanCoarseScorerBase):
    def n_glycan_coarse_score(self, scan: ProcessedScan,
                              glycan_combination: GlycanCombinationRecord,
                              mass_shift: MassShift = Unmodified,
                              peptide_mass: Optional[float] = None) -> Tuple[float, CoarseGlycanMatch]:
        '''Calculates a ranking score from N-glycopeptide stub-glycopeptide fragments.

        This method is derived from the technique used in pGlyco2 [1].

        References
        ----------
        [1] Liu, M.-Q., Zeng, W.-F., Fang, P., Cao, W.-Q., Liu, C., Yan, G.-Q., … Yang, P.-Y. (2017).
            pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality control and
            one-step mass spectrometry for intact glycopeptide identification. Nature Communications,
            8(1), 438. https://doi.org/10.1038/s41467-017-00535-2
        '''
        if peptide_mass is None:
            peptide_mass = (
                scan.precursor_information.neutral_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
        if peptide_mass < 0:
            return -1e6, None
        glycan_match = self._n_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        return score, glycan_match

    def o_glycan_coarse_score(self, scan: ProcessedScan,
                              glycan_combination: GlycanCombinationRecord,
                              mass_shift: MassShift = Unmodified,
                              peptide_mass: Optional[float] = None) -> Tuple[float, CoarseGlycanMatch]:
        '''Calculates a ranking score from O-glycopeptide stub-glycopeptide fragments.

        This method is derived from the technique used in pGlyco2 [1].

        References
        ----------
        [1] Liu, M.-Q., Zeng, W.-F., Fang, P., Cao, W.-Q., Liu, C., Yan, G.-Q., … Yang, P.-Y. (2017).
            pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality control and
            one-step mass spectrometry for intact glycopeptide identification. Nature Communications,
            8(1), 438. https://doi.org/10.1038/s41467-017-00535-2
        '''
        if peptide_mass is None:
            peptide_mass = (
                scan.precursor_information.neutral_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
        if peptide_mass < 0:
            return -1e6, None
        glycan_match = self._o_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        return score, glycan_match

    def gag_coarse_score(self, scan: ProcessedScan,
                         glycan_combination: GlycanCombinationRecord,
                         mass_shift: MassShift = Unmodified,
                         peptide_mass: Optional[float] = None) -> Tuple[float, CoarseGlycanMatch]:
        '''Calculates a ranking score from GAG linker glycopeptide stub-glycopeptide fragments.

        This method is derived from the technique used in pGlyco2 [1].

        References
        ----------
        [1] Liu, M.-Q., Zeng, W.-F., Fang, P., Cao, W.-Q., Liu, C., Yan, G.-Q., … Yang, P.-Y. (2017).
            pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality control and
            one-step mass spectrometry for intact glycopeptide identification. Nature Communications,
            8(1), 438. https://doi.org/10.1038/s41467-017-00535-2
        '''
        if peptide_mass is None:
            peptide_mass = (
                scan.precursor_information.neutral_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
        if peptide_mass < 0:
            return -1e6, None
        glycan_match = self._gag_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        return score, glycan_match

    def match(self, scan, mass_shift=Unmodified, query_mass=None) -> List[GlycanMatchResult]:
        output = []
        if query_mass is None:
            intact_mass = scan.precursor_information.neutral_mass
        else:
            intact_mass = query_mass
        threshold_mass = (intact_mass + 1) - self.minimum_peptide_mass
        for glycan_combination in self.glycan_combination_db:
            # Stop searching when the peptide mass would be below the minimum peptide mass
            if threshold_mass < glycan_combination.dehydrated_mass:
                break
            peptide_mass = (
                intact_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
            best_score = 0
            best_match = None
            type_to_score = {}
            if glycan_combination.is_n_glycan():
                score, match = self.n_glycan_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                type_to_score[GlycanTypes.n_glycan] = (score, match)
                if score > best_score:
                    best_score = score
                    best_match = match
            if glycan_combination.is_o_glycan():
                score, match = self.o_glycan_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                type_to_score[GlycanTypes.o_glycan] = (score, match)
                if score > best_score:
                    best_score = score
                    best_match = match
            if glycan_combination.is_gag_linker():
                score, match = self.gag_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                type_to_score[GlycanTypes.gag_linker] = (score, match)
                if score > best_score:
                    best_score = score
                    best_match = match

            if best_match is not None:
                recalibrated_peptide_mass = best_match.estimate_peptide_mass()
                if recalibrated_peptide_mass > 0:
                    if abs(recalibrated_peptide_mass - peptide_mass) > 0.5:
                        warnings.warn("Re-estimated peptide mass error is large: %f vs %f" % (
                            peptide_mass, recalibrated_peptide_mass))
            else:
                recalibrated_peptide_mass = 0
            result = GlycanMatchResult(
                peptide_mass,
                best_score, best_match, glycan_combination.size, type_to_score, recalibrated_peptide_mass)
            output.append(result)
        output = sorted(output, key=lambda x: x.score, reverse=1)
        return output


class IntervalFilter(Sequence):
    def __init__(self, intervals):
        self.intervals = intervals

    def test(self, mass):
        for i in self.intervals:
            if mass in i:
                return True
        return False

    def __getitem__(self, i):
        return self.intervals[i]

    def __len__(self):
        return len(self.intervals)

    def __call__(self, mass):
        return self.test(mass)

    def compress(self):
        if len(self) == 0:
            return self
        out = []
        last = self[0]
        for interval in self[1:]:
            if interval.overlaps(last):
                last.extend(interval)
            else:
                out.append(last)
                last = interval
        out.append(last)
        self.intervals = out
        return self


try:
    has_c = True
    _IntervalFilter = IntervalFilter
    _CoarseStubGlycopeptideFragment = CoarseStubGlycopeptideFragment
    _CoarseGlycanMatch = CoarseGlycanMatch
    from glycresoft._c.structure.intervals import IntervalFilter
    from glycresoft._c.tandem.core_search import (
        CoarseStubGlycopeptideFragment, CoarseGlycanMatch, GlycanMatchResult,
        GlycanFilteringPeptideMassEstimator_match)

    GlycanFilteringPeptideMassEstimator.match = GlycanFilteringPeptideMassEstimator_match
except ImportError:
    has_c = False


class IndexGlycanCompositionFragment(GlycanCompositionFragment):
    __slots__ = ('index', 'name')

    index: int
    name: str

    def __init__(self, mass, composition, key, is_extended=False):
        self.mass = mass
        self.composition = composition
        self.key = key
        self.is_extended = is_extended
        self._hash_key = -1
        self.index = -1
        self.name = None

    def __reduce__(self):
        return self.__class__, (self.mass, self.composition, self.key, self.is_extended), self.__getstate__()

    def __getstate__(self):
        return {
            "index": self.index,
            "name": self.name,
            "_hash_key": self._hash_key
        }

    def __setstate__(self, state):
        self.index = state['index']
        self.name = state['name']
        self._hash_key = state['_hash_key']


class ComplementFragment(object):
    __slots__ = ('mass', 'keys')

    mass: float
    keys: List[Tuple[IndexGlycanCompositionFragment, int, Any]]

    def __init__(self, mass, keys=None):
        self.mass = mass
        self.keys = keys or []

    def __repr__(self):
        return "{self.__class__.__name__}({self.mass})".format(self=self)


GlycanTypes_n_glycan = GlycanTypes.n_glycan
GlycanTypes_o_glycan = GlycanTypes.o_glycan
GlycanTypes_gag_linker = GlycanTypes.gag_linker


class PartialGlycanSolution(object):
    __slots__ = ("peptide_mass", "score", "core_matches", "fragment_matches", "glycan_index")

    peptide_mass: float
    score: float
    core_matches: Set[int]
    fragment_matches: Set[int]
    glycan_index: int

    def __init__(self, peptide_mass=-1, score=0, core_matches=None, fragment_matches=None, glycan_index=-1):
        if core_matches is None:
            core_matches = set()
        if fragment_matches is None:
            fragment_matches = set()
        self.peptide_mass = peptide_mass
        self.score = score
        self.core_matches = core_matches
        self.fragment_matches = fragment_matches
        self.glycan_index = glycan_index

    def __repr__(self):
        template = ("{self.__class__.__name__}({self.peptide_mass}, {self.score}, "
                    "{self.core_matches}, {self.fragment_matches}, {self.glycan_index})")
        return template.format(self=self)

    @property
    def fragment_match_count(self):
        return len(self.fragment_matches)


class GlycanFragmentIndex(object):
    '''A fast and sparse in-memory fragment ion index for quickly matching peaks against multiple
    glycan composition complements.

    Based upon the complement ion indexing strategy described in [1]_.

    Attributes
    ----------
    members : :class:`list` of :class:`GlycanCombinationRecord`
        The glycan composition combinations in this index.
    unique_fragments : :class:`dict`[:class:`str`, :class:`dict`[:class:`str`, :class:`IndexGlycanCompositionFragment`]]
        An internment table for each glycosylation type mapping fragment name to glycan composition fragments
    fragment_index : :class:`dict`[:class:`int`, :class:`list`[:class:`ComplementFragment`]]
        A sparse index mapping :attr:`resolution` scaled masses to bins. The mass values and binned
        fragments are complements of true fragments.
    counter : int
        A counter to assign each unique fragment a unique integer index.
    fragment_weight : float
        A scoring parameter to weight overall coverage with.
    core_weight : float
        A scoring parameter to weight core motif coverage with.
    resolution : float
        A scaling factor to convert real masses into truncated bin indices.


    References
    ----------
    ..[1] Zeng, W., Cao, W., Liu, M., He, S., & Yang, P. (2021). Precise, Fast and
      Comprehensive Analysis of Intact Glycopeptides and Monosaccharide-Modifications with pGlyco3.
      Bioarxiv. https://doi.org/https://doi.org/10.1101/2021.02.06.430063
    '''

    members: List[GlycanCombinationRecord]
    unique_fragments: DefaultDict[str, Dict[str, IndexGlycanCompositionFragment]]
    fragment_index: DefaultDict[int, List[ComplementFragment]]
    counter: int

    core_weight: float
    fragment_weight: float

    resolution: float
    lower_bound: float
    upper_bound: float


    def __init__(self, members=None, fragment_weight=0.56, core_weight=0.42, resolution=100):
        self.members = members or []
        self.unique_fragments = defaultdict(dict)
        self._fragments = []
        self.fragment_index = defaultdict(list)
        self.counter = 0
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight
        self.lower_bound = float('inf')
        self.upper_bound = 0
        self.resolution = resolution

    def _intern(self, fragment: StubFragment, glycosylation_type: str) -> IndexGlycanCompositionFragment:
        key = str(HashableGlycanComposition(fragment.key))
        try:
            return self.unique_fragments[glycosylation_type][key]
        except KeyError:
            fragment = IndexGlycanCompositionFragment(
                fragment.mass, None, fragment.key, not fragment.is_core)
            self.unique_fragments[glycosylation_type][key] = fragment
            fragment.name = key
            fragment.index = self.counter
            self._fragments.append(fragment)
            self.counter += 1
            return fragment

    def _get_fragments(self, gcr: GlycanCombinationRecord, glycosylation_type: str) -> List[StubFragment]:
        if glycosylation_type == GlycanTypes_n_glycan:
            return gcr.get_n_glycan_fragments()
        elif glycosylation_type == GlycanTypes_o_glycan:
            return gcr.get_o_glycan_fragments()
        elif glycosylation_type == GlycanTypes_gag_linker:
            return gcr.get_gag_linker_glycan_fragments()
        else:
            raise ValueError(glycosylation_type)

    def build(self):
        self.members.sort(key=lambda x: (x.dehydrated_mass, x.id))
        self.fragment_index.clear()
        j = 0
        low = float('inf')
        high = 0
        for i, member in enumerate(self.members):
            mass = member.dehydrated_mass
            n_core = 0
            n_frags = 0
            for glycosylation_type in member.glycan_types:
                for frag in self._get_fragments(member, glycosylation_type):
                    frag = self._intern(frag, glycosylation_type)
                    if not frag.is_extended:
                        n_core += 1
                    n_frags += 1
                    d = mass - frag.mass
                    if d < 0:
                        d = 0
                    if high < d:
                        high = d
                    if low > d:
                        low = d
                    key = int(d * self.resolution)
                    for comp in self.fragment_index[key]:
                        if abs(comp.mass - d) <= 1e-3:
                            comp.keys.append((frag, i, glycosylation_type))
                            break
                    else:
                        self.fragment_index[key].append(ComplementFragment(d, [(frag, i, glycosylation_type)]))
                        j += 1

                member.fragment_set_properties[glycosylation_type] = (n_core, n_frags)
            member.clear()
        self.lower_bound = low
        self.upper_bound = high
        return j

    def _match_fragments(self, delta_mass: float, peak: DeconvolutedPeak,
                         error_tolerance: float, result: Dict[int, Dict[str, PartialGlycanSolution]]):
        key = int(delta_mass * self.resolution)
        width = int(peak.neutral_mass * error_tolerance * self.resolution) + 1
        for off in range(-width, width + 1):
            for comp_frag in self.fragment_index[key + off]:
                if abs(comp_frag.mass - delta_mass) / peak.neutral_mass <= error_tolerance:
                    for frag, i, glycosylation_type in comp_frag.keys:
                        sol = result[i][glycosylation_type]
                        sol.score += np.log10(peak.intensity) * (
                            1 - ((abs(delta_mass - comp_frag.mass) / peak.neutral_mass) / error_tolerance) ** 4)
                        if not frag.is_extended:
                            sol.core_matches.add(frag.index)
                        sol.fragment_matches.add(frag.index)

    def match(self, scan: ProcessedScan, error_tolerance: float=1e-5,
              mass_shift: MassShift=Unmodified, query_mass: Optional[float]=None) -> List[PartialGlycanSolution]:
        if query_mass is None:
            precursor_mass: float = scan.precursor_information.neutral_mass
        else:
            precursor_mass = query_mass

        mass_shift_mass = mass_shift.mass
        mass_shift_tandem_mass = mass_shift.tandem_mass

        result: DefaultDict[int, DefaultDict[str, PartialGlycanSolution]] = defaultdict(
            lambda: defaultdict(PartialGlycanSolution))
        for peak in scan.deconvoluted_peak_set:
            d = (precursor_mass - peak.neutral_mass - mass_shift_mass)
            self._match_fragments(d, peak, error_tolerance, result)
            if mass_shift_tandem_mass != 0:
                d = (precursor_mass - peak.neutral_mass - mass_shift_mass + mass_shift_tandem_mass)
                self._match_fragments(d, peak, error_tolerance, result)

        out = []
        for i, glycosylation_type_to_solutions in result.items():
            rec = self.members[i]
            best_score = -float('inf')
            best_solution = None
            for glycosylation_type, sol in glycosylation_type_to_solutions.items():
                n_core, n_frag = rec.fragment_set_properties[glycosylation_type]
                coverage = (len(sol.core_matches) * 1.0 / n_core) ** self.core_weight * (
                    len(sol.fragment_matches) * 1.0 / n_frag) ** self.fragment_weight
                sol.score *= coverage
                sol.peptide_mass = precursor_mass - rec.dehydrated_mass - mass_shift_mass
                sol.glycan_index = i
                if best_score < sol.score and sol.peptide_mass > 0:
                    best_score = sol.score
                    best_solution = sol
            if best_solution is not None:
                out.append(sol)

        out.sort(key=lambda x: x.score, reverse=True)
        return out


try:
    _GlycanFragmentIndex = GlycanFragmentIndex
    _IndexGlycanCompositionFragment = IndexGlycanCompositionFragment
    _ComplementFragment = ComplementFragment
    _PartialGlycanSolution = PartialGlycanSolution
    from glycresoft._c.tandem.core_search import (
        GlycanFragmentIndex,
        IndexGlycanCompositionFragment, ComplementFragment, PartialGlycanSolution)
except ImportError:
    pass


class _adapt(object):
    def __init__(self, *args, **kwargs):
        pass


class IndexedGlycanFilteringPeptideMassEstimator(GlycanFilteringPeptideMassEstimatorBase[PartialGlycanSolution],
                                                 _adapt):
    """
    An inverted complement search index for quickly searching for estimating peptide backbone masses
    and glycans to go with them.

    Examples
    --------

    .. code-block:: python

        from glycresoft.tandem.glycopeptide.core_search import (
            GlycanCombinationRecord, IndexedGlycanFilteringPeptideMassEstimator)
        from glycresoft import serialize

        def make_glycan_index(db_path, glycan_hypothesis_id, product_error_tolerance: float=2e-5):
            # Or any object that has a `session` attribute could be used instead of making one here
            db_conn = serialize.DatabaseBoundOperation(db_path)
            # This is the set of *all* glycan combinations for that hypothesis.
            glycan_combinations = GlycanCombinationRecord.from_hypothesis(db_conn, glycan_hypothesis_id)
            search_index = IndexedGlycanFilteringPeptideMassEstimator(
                glycan_combinations, product_error_tolerance=product_error_tolerance)
            return search_index

        db_path = ... # magically get a database path
        scan = ... # magically get a `ProcessedScan` to demonstrate on

        search_index = make_glycan_index(db_path, 1)
        peptide_backbone_mass_candidates = search_index.match(scan)
        for rec in peptide_backbone_mass_candidates:
            candidate_glycans = search_index.glycan_for_peptide_mass(scan, rec.peptide_mass)
            print(rec, candidate_glycans)
    """
    product_error_tolerance: float
    fragment_weight: float
    core_weight: float

    def __init__(self, glycan_combination_db, product_error_tolerance=1e-5,
                 fragment_weight=0.56, core_weight=0.42, minimum_peptide_mass=500.0,
                 use_denovo_motif=False, components=None,
                 use_recalibrated_peptide_mass=False):
        super(IndexedGlycanFilteringPeptideMassEstimator, self).__init__(
            glycan_combination_db, product_error_tolerance, fragment_weight, core_weight,
            minimum_peptide_mass, use_denovo_motif, components, use_recalibrated_peptide_mass)
        self.product_error_tolerance = product_error_tolerance
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight
        self.index = GlycanFragmentIndex(
            self.glycan_combination_db, fragment_weight=fragment_weight, core_weight=core_weight)
        self.index.build()

    def match(self, scan: ProcessedScan, mass_shift: MassShift=Unmodified,
              query_mass: Optional[float]=None) -> List[PartialGlycanSolution]:
        return self.index.match(
            scan,
            error_tolerance=self.product_error_tolerance,
            mass_shift=mass_shift,
            query_mass=query_mass)
