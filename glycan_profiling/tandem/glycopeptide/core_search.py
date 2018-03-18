import logging

from collections import Sequence, defaultdict

import numpy as np

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue
from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy

from glycan_profiling.serialize import GlycanCombination
from glycan_profiling.structure import SpectrumGraph
from glycan_profiling.database.disk_backed_database import PPMQueryInterval
from glycan_profiling.chromatogram_tree import Unmodified

logger = logging.getLogger("glycresoft.tandem")


hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
xylose = FrozenMonosaccharideResidue.from_iupac_lite("Xyl")
fucose = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")
# neuac = FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")


default_components = (hexnac, hexose, xylose, fucose,)


class Path(object):
    def __init__(self, edge_list):
        self.transitions = edge_list
        self.total_signal = self._total_signal()
        self.start_mass = self[0].start.neutral_mass
        self.end_mass = self[-1].end.neutral_mass

    def __iter__(self):
        return iter(self.transitions)

    def __getitem__(self, i):
        return self.transitions[i]

    def __len__(self):
        return len(self.transitions)

    def _total_signal(self):
        total = 0
        for edge in self:
            total += edge.end.intensity
        total += self[0].start.intensity
        return total

    def __repr__(self):
        return "%s(%s, %0.4e, %f, %f)" % (
            self.__class__.__name__,
            '->'.join(str(e.annotation) for e in self),
            self.total_signal, self.start_mass, self.end_mass
        )


class PathSet(Sequence):
    def __init__(self, paths, ordered=False):
        self.paths = (sorted(paths, key=lambda x: x.start_mass)
                      if not ordered else paths)

    def __getitem__(self, i):
        return self.paths[i]

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.paths)[1:-1])

    def _repr_pretty_(self, p, cycle):
        with p.group(9, "%s([" % self.__class__.__name__, "])"):
            for i, path in enumerate(self):
                if i:
                    p.text(",")
                    p.breakable()
                p.pretty(path)


class CoreMotifFinder(object):
    def __init__(self, components=None, product_error_tolerance=1e-5):
        if components is None:
            components = default_components
        self.components = components
        self.product_error_tolerance = product_error_tolerance

    def _find_edges(self, scan, mass_shift=Unmodified):
        graph = SpectrumGraph()
        has_tandem_shift = abs(mass_shift.tandem_mass) > 0
        for peak in scan.deconvoluted_peak_set:
            for component in self.components:
                for other_peak in scan.deconvoluted_peak_set.all_peaks_for(
                        peak.neutral_mass + component.mass(), self.product_error_tolerance):
                    graph.add(peak, other_peak, component)
                if has_tandem_shift:
                    for other_peak in scan.deconvoluted_peak_set.all_peaks_for(
                            peak.neutral_mass + component.mass() + mass_shift.tandem_mass,
                            self.product_error_tolerance):
                        graph.add(peak, other_peak, component)
        return graph

    def _init_paths(self, graph, limit=1000):
        paths = []
        min_start_mass = max(c.mass() for c in self.components) + 1
        for path in graph.longest_paths(limit=limit):
            path = Path(path)
            if path.start_mass < min_start_mass:
                continue
            paths.append(path)
        return paths

    def _aggregate_paths(self, paths):
        groups = defaultdict(list)
        for path in paths:
            label = tuple(p.annotation for p in path)
            groups[label].append(path)
        return groups

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

        # TODO: split the different motif masses up according to core type efficiently
        # but for now just lump them all together
        for path in n_linked_paths:
            peptide_masses.append(path.start_mass)
        for path in o_linked_paths:
            peptide_masses.append(path.start_mass)
        for path in gag_linker_paths:
            peptide_masses.append(path.start_mass)
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


class GlycanCombinationRecord(object):
    __slots__ = ['dehydrated_mass', 'composition', 'count', 'glycan_types', "_fragment_cache"]

    @classmethod
    def from_combination(cls, combination):
        inst = cls(
            dehydrated_mass=combination.dehydrated_mass(),
            composition=combination.convert(),
            count=combination.count,
            # TODO: make this meaningful downstream
            # glycan_types=combination.component_classes
            glycan_types=[],
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

    def __init__(self, dehydrated_mass, composition, count, glycan_types):
        self.dehydrated_mass = dehydrated_mass
        self.composition = composition
        self.count = count
        self.glycan_types = glycan_types
        self._fragment_cache = None

    def __reduce__(self):
        return GlycanCombinationRecord, (self.dehydrated_mass, self.composition, self.count, self.glycan_types)

    def __repr__(self):
        return "GlycanCombinationRecord(%s, %d)" % (self.composition, self.count)


class GlycanFilteringPeptideMassEstimator(object):
    def __init__(self, glycan_combination_db, product_error_tolerance=1e-5,
                 alpha=0.56, beta=0.42, components=None):
        if not isinstance(glycan_combination_db[0], GlycanCombinationRecord):
            glycan_combination_db = [GlycanCombinationRecord.from_combination(gc)
                                     for gc in glycan_combination_db]
        self.motif_finder = CoreMotifFinder(components, product_error_tolerance)
        self.product_error_tolerance = product_error_tolerance
        self.glycan_combination_db = glycan_combination_db
        self.fragment_generator = StubGlycopeptideStrategy(None, extended=True)
        self.alpha = alpha
        self.beta = beta

    def _n_glycan_match_stubs(self, scan, peptide_mass, glycan_combination, mass_shift=Unmodified):
        glycan_composition = glycan_combination.composition
        if glycan_combination._fragment_cache is None:
            shifts = self.fragment_generator.n_glycan_composition_fragments(glycan_composition, 1, 0)
            glycan_combination._fragment_cache = shifts
        else:
            shifts = glycan_combination._fragment_cache
        fragment_matches = []
        core_matched = 0.0
        core_theoretical = 0.0
        has_tandem_shift = abs(mass_shift.tandem_mass) > 0
        for shift in shifts:
            if shift["key"].get('HexNAc', 0) <= 2 and shift["key"].get("Hex", 0) <= 3:
                is_core = True
                core_theoretical += 1
            target_mass = shift['mass'] + peptide_mass
            hits = scan.deconvoluted_peak_set.all_peaks_for(target_mass, self.product_error_tolerance)
            if hits:
                if is_core:
                    core_matched += 1
                fragment_matches.append((shift['key'], target_mass, hits))
            if has_tandem_shift:
                shifted_mass = target_mass + mass_shift.tandem_mass
                hits = scan.deconvoluted_peak_set.all_peaks_for(
                    shifted_mass, self.product_error_tolerance)
                if hits:
                    if is_core:
                        core_matched += 1
                    fragment_matches.append((shift['key'], shifted_mass, hits))

        return fragment_matches, float(len(fragment_matches)), float(len(shifts)), core_matched, core_theoretical

    def n_glycan_coarse_score(self, scan, glycan_combination, mass_shift=Unmodified):
        peptide_mass = (scan.precursor_information.neutral_mass - glycan_combination.dehydrated_mass)
        matched_fragments, n_matched, n_theoretical, core_matched, core_theoretical = self._n_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift=mass_shift)
        ratio_fragments = (n_matched / n_theoretical)
        ratio_core = core_matched / core_theoretical
        score = 0
        for key, mass, matches in matched_fragments:
            for peak in matches:
                score += np.log(peak.intensity) * (
                    1 - (np.abs(peak.neutral_mass - mass) / mass) ** 4) * (
                    ratio_fragments ** self.alpha) * (ratio_core ** self.beta)
        return peptide_mass, score

    def _estimate_peptide_mass(self, scan, mass_shift=Unmodified):
        output = []
        for glycan_combination in self.glycan_combination_db:
            result = self.n_glycan_coarse_score(scan, glycan_combination, mass_shift=mass_shift)
            output.append(result)
        output.sort(key=lambda x: x[1], reverse=1)
        return output

    def estimate_peptide_mass(self, scan, topn=100, mass_shift=Unmodified):
        out = self._estimate_peptide_mass(scan, mass_shift=mass_shift)
        out = out[:topn]
        return [x[0] for x in out]

    def build_peptide_filter(self, scan, error_tolerance=1e-5, mass_shift=Unmodified):
        peptide_masses = self.estimate_peptide_mass(scan, mass_shift=mass_shift)
        peptide_masses = [PPMQueryInterval(p, error_tolerance) for p in peptide_masses]
        path_masses = self.motif_finder.build_peptide_filter(scan, error_tolerance, mass_shift=mass_shift)
        peptide_masses.extend(path_masses)
        peptide_masses.sort(key=lambda x: x.center)

        if len(peptide_masses) == 0:
            return IntervalFilter([])
        out = []
        last = peptide_masses[0]
        for interval in peptide_masses[1:]:
            if interval.overlaps(last):
                last.extend(interval)
            else:
                out.append(last)
                last = interval
        out.append(last)
        return IntervalFilter(out)


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


try:
    has_c = True
    _IntervalFilter = IntervalFilter

    from glycan_profiling._c.structure.intervals import IntervalFilter
except ImportError:
    has_c = False
