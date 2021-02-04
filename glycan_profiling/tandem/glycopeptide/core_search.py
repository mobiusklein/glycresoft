# -*- coding: utf-8 -*-

import logging
import warnings
from collections import namedtuple

try:
    from collections import Sequence
except ImportError:
    from collections.abc import Sequence

import numpy as np

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, HashableGlycanComposition
from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy, _AccumulatorBag

from glycan_profiling.serialize import GlycanCombination, GlycanTypes
from glycan_profiling.database.disk_backed_database import PPMQueryInterval
from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.structure.denovo import MassWrapper, PathSet, PathFinder

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
    def __init__(self, components=None, product_error_tolerance=1e-5, minimum_peptide_mass=350.): # pylint: disable=super-init-not-called
        if components is None:
            components = default_components
        self.components = list(map(MassWrapper, components))
        self.product_error_tolerance = product_error_tolerance
        self.minimum_peptide_mass = minimum_peptide_mass

    def find_n_linked_core(self, groups, min_size=1):
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

    def find_o_linked_core(self, groups, min_size=1):
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

    def find_gag_linker_core(self, groups, min_size=1):
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

    def estimate_peptide_mass(self, scan, topn=100, mass_shift=Unmodified, query_mass=None):
        graph = self._find_edges(scan, mass_shift=mass_shift)
        paths = self._init_paths(graph)
        groups = self._aggregate_paths(paths)

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
            peptide_masses.append(path.start_mass)
            if has_tandem_shift:
                peptide_masses.append(path.start_mass - mass_shift.tandem_mass)
        for path in o_linked_paths:
            if path.start_mass < self.minimum_peptide_mass:
                continue
            peptide_masses.append(path.start_mass)
            if has_tandem_shift:
                peptide_masses.append(path.start_mass - mass_shift.tandem_mass)
        for path in gag_linker_paths:
            if path.start_mass < self.minimum_peptide_mass:
                continue
            peptide_masses.append(path.start_mass)
            if has_tandem_shift:
                peptide_masses.append(path.start_mass - mass_shift.tandem_mass)
        peptide_masses.sort()
        return peptide_masses

    def build_peptide_filter(self, scan, error_tolerance=1e-5, mass_shift=Unmodified, query_mass=none):
        peptide_masses = self.estimate_peptide_mass(
            scan, mass_shift=mass_shift, query_mass=query_mass)

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
                 'size', "_fragment_cache", "internal_size_approximation", "_hash"]

    def is_n_glycan(self):
        return GlycanTypes.n_glycan in self.glycan_types

    def is_o_glycan(self):
        return GlycanTypes.o_glycan in self.glycan_types

    def is_gag_linker(self):
        return GlycanTypes.gag_linker in self.glycan_types

    def get_n_glycan_fragments(self):
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

    def get_o_glycan_fragments(self):
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

    def get_gag_linker_glycan_fragments(self):
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

try:
    from glycan_profiling._c.tandem.core_search import GlycanCombinationRecordBase
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

    __slots__ = []

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
        return GlycanCombinationRecord, (self.id, self.dehydrated_mass, self.composition, self.count, self.glycan_types)

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
    from glycan_profiling._c.tandem.core_search import GlycanCoarseScorerBase
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


class GlycanFilteringPeptideMassEstimator(GlycanCoarseScorerBase):
    def __init__(self, glycan_combination_db, product_error_tolerance=1e-5,
                 fragment_weight=0.56, core_weight=0.42, minimum_peptide_mass=500.0,
                 use_denovo_motif=False, components=None,
                 use_recalibrated_peptide_mass=False):
        if not isinstance(glycan_combination_db[0], GlycanCombinationRecord):
            glycan_combination_db = [GlycanCombinationRecord.from_combination(gc)
                                     for gc in glycan_combination_db]
        self.use_denovo_motif = use_denovo_motif
        self.motif_finder = CoreMotifFinder(components, product_error_tolerance)
        self.glycan_combination_db = sorted(glycan_combination_db, key=lambda x: (x.dehydrated_mass, x.id))
        self.minimum_peptide_mass = minimum_peptide_mass
        self.use_recalibrated_peptide_mass = use_recalibrated_peptide_mass
        super(GlycanFilteringPeptideMassEstimator, self).__init__(
            product_error_tolerance, fragment_weight, core_weight)

    def n_glycan_coarse_score(self, scan, glycan_combination, mass_shift=Unmodified, peptide_mass=None):
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

    def o_glycan_coarse_score(self, scan, glycan_combination, mass_shift=Unmodified, peptide_mass=None):
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

    def gag_coarse_score(self, scan, glycan_combination, mass_shift=Unmodified, peptide_mass=None):
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

    def match(self, scan, mass_shift=Unmodified, query_mass=None):
        output = []
        if query_mass is None:
            intact_mass = scan.precursor_information.neutral_mass
        else:
            intact_mass = query_mass
        threshold_mass = (intact_mass + 1) - self.minimum_peptide_mass
        last_peptide_mass = 0
        for glycan_combination in self.glycan_combination_db:
            # Stop searching when the peptide mass would be below the minimum peptide mass
            if threshold_mass < glycan_combination.dehydrated_mass:
                break
            peptide_mass = (
                intact_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
            last_peptide_mass = peptide_mass
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

    def estimate_peptide_mass(self, scan, topn=150, threshold=-1, min_fragments=0, mass_shift=Unmodified,
                              simplify=True, query_mass=None):
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
        out = [x for x in out if x.score > threshold and x.fragment_match_count >= min_fragments]
        groups = group_by_score(out)
        out = flatten(groups[:topn])
        if simplify:
            return [x.peptide_mass for x in out]
        return out

    def glycan_for_peptide_mass(self, scan, peptide_mass):
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

    def build_peptide_filter(self, scan, error_tolerance=None, mass_shift=Unmodified, query_mass=None):
        if error_tolerance is None:
            error_tolerance = self.product_error_tolerance
        peptide_masses = self.estimate_peptide_mass(
            scan, mass_shift=mass_shift, query_mass=query_mass)
        peptide_masses = [PPMQueryInterval(p, error_tolerance) for p in peptide_masses]
        if self.use_denovo_motif:
            path_masses = self.motif_finder.build_peptide_filter(scan, error_tolerance, mass_shift=mass_shift)
            peptide_masses.extend(path_masses)
        peptide_masses.sort(key=lambda x: x.center)

        if len(peptide_masses) == 0:
            return IntervalFilter([])
        out = IntervalFilter(peptide_masses)
        out.compress()
        return out


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
    from glycan_profiling._c.structure.intervals import IntervalFilter
    from glycan_profiling._c.tandem.core_search import (
        CoarseStubGlycopeptideFragment, CoarseGlycanMatch, GlycanMatchResult,
        GlycanFilteringPeptideMassEstimator_match)

    GlycanFilteringPeptideMassEstimator.match = GlycanFilteringPeptideMassEstimator_match
except ImportError:
    has_c = False
