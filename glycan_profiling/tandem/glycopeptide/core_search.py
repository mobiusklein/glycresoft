# -*- coding: utf-8 -*-

import logging

try:
    from collections import Sequence
except ImportError:
    from collections.abc import Sequence

import numpy as np

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue
from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy, _AccumulatorBag

from glycan_profiling.serialize import GlycanCombination, GlycanTypes
from glycan_profiling.database.disk_backed_database import PPMQueryInterval
from glycan_profiling.chromatogram_tree import Unmodified
from glycan_profiling.structure.denovo import MassWrapper, PathSet, PathFinder

logger = logging.getLogger("glycresoft.tandem")


hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
xylose = FrozenMonosaccharideResidue.from_iupac_lite("Xyl")
fucose = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")
# neuac = FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")


class maxdict(dict):
    '''A dictionary that only updates a key's value if it exceeds the
    current value'''
    def __init__(self, base=None, **kwargs):
        dict.__init__(self, **kwargs)
        if base is not None:
            self.update(base)

    def __setitem__(self, key, value):
        current_value = self[key]
        if value < current_value:
            return
        super(maxdict, self).__setitem__(key, value)

    def __missing__(self, key):
        super(maxdict, self).__setitem__(key, 0)

    def __repr__(self):
        return "maxdict(%r)" % (dict(self), )

    @classmethod
    def fromsequence(cls, seq):
        self = cls()
        for k, v in seq:
            self[k] = v
        return self


default_components = (hexnac, hexose, xylose, fucose,)


class CoreMotifFinder(PathFinder):
    def __init__(self, components=None, product_error_tolerance=1e-5, minimum_peptide_mass=350.):
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

    def estimate_peptide_mass(self, scan, topn=100, mass_shift=Unmodified):
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

    def build_peptide_filter(self, scan, error_tolerance=1e-5, mass_shift=Unmodified):
        peptide_masses = self.estimate_peptide_mass(scan, mass_shift=mass_shift)

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


class GlycanCombinationRecord(object):
    """Represent a glycan combination compactly in memory

    Attributes
    ----------
    composition : :class:`~.HashsableGlycanComposition`
        The glycan combination's composition in monosaccharide units
    count : int
        The number of distinct glycans this combination contains
    dehydrated_mass : float
        The total mass shift applied to a peptide when this combination is attached
        to it
    glycan_types : list
        The types of glycans combined to make this entity
    """

    __slots__ = ['id', 'dehydrated_mass', 'composition', 'count', 'glycan_types', "_fragment_cache"]

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
            GlycanCombination.dehydrated_mass())
        candidates = query.all()
        out = []
        for candidate in candidates:
            out.append(cls.from_combination(candidate))
        return out

    def __init__(self, id, dehydrated_mass, composition, count, glycan_types):
        self.id = id
        self.dehydrated_mass = dehydrated_mass
        self.composition = composition
        self.count = count
        self.glycan_types = glycan_types
        self._fragment_cache = dict()

    def __eq__(self, other):
        return (self.composition == other.composition) and (self.count == other.count) and (
            self.glycan_types == other.glycan_types)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.composition)

    def is_n_glycan(self):
        return GlycanTypes.n_glycan in self.glycan_types

    def is_o_glycan(self):
        return GlycanTypes.o_glycan in self.glycan_types

    def is_gag_linker(self):
        return GlycanTypes.gag_linker in self.glycan_types

    def __reduce__(self):
        return GlycanCombinationRecord, (self.id, self.dehydrated_mass, self.composition, self.count, self.glycan_types)

    def __repr__(self):
        return "GlycanCombinationRecord(%s, %d)" % (self.composition, self.count)

    def get_n_glycan_fragments(self):
        if GlycanTypes.n_glycan not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.n_glycan_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                shift['key'] = _AccumulatorBag(shift['key'])
                if shift["key"]['HexNAc'] <= 2 and shift["key"]["Hex"] <= 3:
                    shift['is_core'] = True
                else:
                    shift['is_core'] = False
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], shift['is_core']))
            self._fragment_cache[GlycanTypes.n_glycan] = sorted(set(fragment_structs))
            return self._fragment_cache[GlycanTypes.n_glycan]
        else:
            return self._fragment_cache[GlycanTypes.n_glycan]

    def get_o_glycan_fragments(self):
        if GlycanTypes.o_glycan not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.o_glycan_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                shift['key'] = _AccumulatorBag(shift['key'])
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            self._fragment_cache[GlycanTypes.o_glycan] = sorted(set(fragment_structs))
            return self._fragment_cache[GlycanTypes.o_glycan]
        else:
            return self._fragment_cache[GlycanTypes.o_glycan]

    def get_gag_linker_glycan_fragments(self):
        if GlycanTypes.gag_linker not in self._fragment_cache:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.gag_linker_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                shift['key'] = _AccumulatorBag(shift['key'])
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            self._fragment_cache[GlycanTypes.gag_linker] = sorted(set(fragment_structs))
            return self._fragment_cache[GlycanTypes.gag_linker]
        else:
            return self._fragment_cache[GlycanTypes.gag_linker]


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
                    fragment_matches.append((shift.key, shifted_mass, hits))

        return CoarseGlycanMatch(
            fragment_matches, float(len(fragment_matches)), float(len(shifts)), core_matched, core_theoretical)

    def _calculate_score(self, matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical):
        ratio_fragments = (n_matched / n_theoretical)
        ratio_core = core_matched / core_theoretical
        coverage = (ratio_fragments ** self.fragment_weight) * (ratio_core ** self.core_weight)
        score = 0
        for key, mass, matches in matched_fragments:
            for peak in matches:
                score += np.log(peak.intensity) * (1 - (np.abs(peak.neutral_mass - mass) / mass) ** 4) * coverage
        return score


try:
    _GlycanCoarseScorerBase = GlycanCoarseScorerBase
    from glycan_profiling._c.tandem.core_search import GlycanCoarseScorerBase
except ImportError:
    pass


class GlycanFilteringPeptideMassEstimator(GlycanCoarseScorerBase):
    def __init__(self, glycan_combination_db, product_error_tolerance=1e-5,
                 fragment_weight=0.56, core_weight=0.42, minimum_peptide_mass=500.0,
                 use_denovo_motif=False, components=None):
        if not isinstance(glycan_combination_db[0], GlycanCombinationRecord):
            glycan_combination_db = [GlycanCombinationRecord.from_combination(gc)
                                     for gc in glycan_combination_db]
        self.use_denovo_motif = use_denovo_motif
        self.motif_finder = CoreMotifFinder(components, product_error_tolerance)
        self.glycan_combination_db = sorted(glycan_combination_db, key=lambda x: x.dehydrated_mass)
        self.minimum_peptide_mass = minimum_peptide_mass
        super(GlycanFilteringPeptideMassEstimator, self).__init__(
            product_error_tolerance, fragment_weight, core_weight)

    def _n_glycan_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_n_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    def _o_glycan_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_o_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    def _gag_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_gag_linker_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

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
            return peptide_mass, -1e6
        matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical = self._n_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical)
        return score

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
            return peptide_mass, -1e6
        matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical = self._o_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical)
        return score

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
            return peptide_mass, -1e6
        matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical = self._gag_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical)
        return score

    def _estimate_peptide_mass(self, scan, mass_shift=Unmodified):
        output = []
        intact_mass = scan.precursor_information.neutral_mass
        threshold_mass = (intact_mass + 1) - self.minimum_peptide_mass
        for glycan_combination in self.glycan_combination_db:
            # Stop searching when the peptide mass would be below the minimum peptide
            # mass.
            if threshold_mass < glycan_combination.dehydrated_mass:
                break
            peptide_mass = (
                intact_mass - glycan_combination.dehydrated_mass
            ) - mass_shift.mass
            best_score = 0
            if glycan_combination.is_n_glycan():
                score = self.n_glycan_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                if score > best_score:
                    best_score = score
            if glycan_combination.is_o_glycan():
                score = self.o_glycan_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                if score > best_score:
                    best_score = score
            if glycan_combination.is_gag_linker():
                score = self.gag_coarse_score(
                    scan, glycan_combination, mass_shift=mass_shift, peptide_mass=peptide_mass)
                if score > best_score:
                    best_score = score
            output.append((peptide_mass, best_score))
        output = sorted(output, key=lambda x: x[1], reverse=1)
        return output

    def estimate_peptide_mass(self, scan, topn=150, mass_shift=Unmodified):
        out = self._estimate_peptide_mass(scan, mass_shift=mass_shift)
        out = out[:topn]
        return [x[0] for x in out]

    def glycan_for_peptide_mass(self, scan, peptide_mass):
        matches = []
        glycan_mass = scan.precursor_information.neutral_mass - peptide_mass
        for glycan_record in self.glycan_combination_db:
            if abs(glycan_record.dehydrated_mass - glycan_mass) / glycan_mass < self.product_error_tolerance:
                matches.append(glycan_record)
        return matches

    def build_peptide_filter(self, scan, error_tolerance=1e-5, mass_shift=Unmodified):
        peptide_masses = self.estimate_peptide_mass(scan, mass_shift=mass_shift)
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
    from glycan_profiling._c.tandem.core_search import (CoarseStubGlycopeptideFragment, CoarseGlycanMatch)
except ImportError:
    has_c = False
