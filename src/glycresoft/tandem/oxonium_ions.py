from collections import defaultdict
from typing import Any, DefaultDict, Dict, FrozenSet, Iterable, List, Tuple, Union, MutableSequence
from statistics import median

from dataclasses import dataclass, field

from glycopeptidepy import PeptideSequence
from glycopeptidepy.structure.fragment import SimpleFragment

from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, Composition

from glycresoft.tandem.glycopeptide.core_search import GlycanCombinationRecord

from ms_deisotope.peak_set import DeconvolutedPeakSet


class _mass_wrapper(object):

    def __init__(self, mass, annotation=None):
        self.value = mass
        self.annotation = annotation if annotation is not None else mass

    def mass(self, *args, **kwargs):
        return self.value

    def __repr__(self):
        return "MassWrapper(%f, %s)" % (self.value, self.annotation)


_hexnac = FrozenMonosaccharideResidue.from_iupac_lite("HexNAc")
_hexose = FrozenMonosaccharideResidue.from_iupac_lite("Hex")
_neuac = FrozenMonosaccharideResidue.from_iupac_lite('NeuAc')
_neugc = FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")
_water = Composition("H2O").mass

_standard_oxonium_ions = [
    _hexnac,
    _hexose,
    _mass_wrapper(_hexose.mass() - _water),
    _neuac,
    _mass_wrapper(_neuac.mass() - _water),
    FrozenMonosaccharideResidue.from_iupac_lite("Fuc"),
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - _water),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]

standard_oxonium_ions = _standard_oxonium_ions[:]

_gscore_oxonium_ions = [
    _hexnac,
    _mass_wrapper(_hexnac.mass() - Composition("C2H6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("CH6O3").mass),
    _mass_wrapper(_hexnac.mass() - Composition("C2H4O2").mass),
    _mass_wrapper(_hexnac.mass() - _water),
    _mass_wrapper(_hexnac.mass() - Composition("H4O2").mass)
]


class OxoniumIonScanner(object):

    def __init__(self, ions_to_search=None):
        if ions_to_search is None:
            ions_to_search = _standard_oxonium_ions
        self.ions_to_search = ions_to_search

    def scan(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        matches = []
        for ion in self.ions_to_search:
            if ion.mass() < minimum_mass:
                continue
            match = peak_list.has_peak(
                ion.mass(charge=charge), error_tolerance)
            if match is not None:
                matches.append(match)
        return matches

    def ratio(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        try:
            maximum = max(p.intensity for p in peak_list)
        except ValueError:
            return 0
        n = len([i for i in self.ions_to_search if i.mass() > minimum_mass])
        if n == 0:
            return 0
        oxonium = sum(
            p.intensity / maximum for p in self.scan(
                peak_list, charge, error_tolerance, minimum_mass))
        return oxonium / n

    def __call__(self, peak_list, charge=0, error_tolerance=2e-5, minimum_mass=0):
        return self.ratio(peak_list, charge, error_tolerance, minimum_mass)


oxonium_detector = OxoniumIonScanner()
gscore_scanner = OxoniumIonScanner(_gscore_oxonium_ions)


class SignatureSpecification(object):
    __slots__ = ('components', 'masses', '_hash', '_is_compound')

    def __init__(self, components, masses):
        self.components = tuple(
            FrozenMonosaccharideResidue.from_iupac_lite(k) for k in components)
        self.masses = tuple(masses)
        self._hash = hash(self.components)
        self._is_compound = len(self.masses) > 1

    def __getitem__(self, i):
        return self.components[i]

    def __iter__(self):
        return iter(self.components)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.components == other.components

    def __repr__(self):
        return "{self.__class__.__name__}({self.components}, {self.masses})".format(self=self)

    def is_expected(self, glycan_composition):
        is_expected = glycan_composition._getitem_fast(self.components[0]) != 0
        if is_expected and self._is_compound:
            is_expected = all(glycan_composition._getitem_fast(
                k) != 0 for k in self.components)
        return is_expected

    def count_of(self, glycan_composition):
        limit = float('inf')
        for component in self:
            cnt = glycan_composition._getitem_fast(component)
            if cnt < limit:
                limit = cnt
        return limit

    def peak_of(self, spectrum, error_tolerance):
        best_peak = None
        best_signal = -1
        for mass in self.masses:
            peaks = spectrum.all_peaks_for(mass, error_tolerance)
            for peak in peaks:
                if peak.intensity > best_signal:
                    best_peak = peak
                    best_signal = peak.intensity
        return best_peak


try:
    _has_c = True
    from glycresoft._c.tandem.oxonium_ions import SignatureSpecification
except ImportError:
    _has_c = False


single_signatures = {
    SignatureSpecification((str(_neuac), ), [
        _neuac.mass(),
        _neuac.mass() - _water
    ]): 0.5,
    SignatureSpecification((str(_neugc), ), [
        _neugc.mass(),
        _neugc.mass() - _water
    ]): 0.5,
}


compound_signatures = {
    SignatureSpecification(('@phosphate', 'Hex'), [
        242.01915393925,
        224.00858925555,
    ]): 0.5,
    SignatureSpecification(("@acetyl", "NeuAc"), [
        333.10598119017,
        315.09541650647
    ]): 0.5
}


class OxoniumIndex(object):
    '''An index for quickly matching all oxonium ions against a spectrum and efficiently mapping them
    back to individual glycan compositions.
    '''
    fragments: List[SimpleFragment]
    fragment_index: DefaultDict[str, List[int]]
    glycan_to_index: Dict[str, int]
    index_to_glycan: Dict[int, Union[str, List[str]]]

    def __init__(self, fragments=None, fragment_index=None, glycan_to_index=None):
        self.fragments = fragments or []
        self.fragment_index = defaultdict(list, fragment_index or {})
        self.glycan_to_index = glycan_to_index or {}
        self.index_to_glycan = {v: k for k, v in self.glycan_to_index.items()}

    def _make_glycopeptide_stub(self, glycan_composition) -> PeptideSequence:
        p = PeptideSequence("P%s" % glycan_composition)
        return p

    def build_index(self, glycan_composition_records: List[GlycanCombinationRecord], **kwargs):
        fragments = {}
        fragment_index = defaultdict(list)
        glycan_index = {}
        for gc_rec in glycan_composition_records:
            glycan_index[gc_rec.composition] = gc_rec.id

            p = self._make_glycopeptide_stub(gc_rec.composition)
            for frag in p.glycan_fragments(**kwargs):
                fragments[frag.name] = frag
                fragment_index[frag.name].append(gc_rec.id)

        self.glycan_to_index = glycan_index
        self.fragment_index = fragment_index
        self.fragments = sorted(fragments.values(), key=lambda x: x.mass)
        self.index_to_glycan = {v: k for k, v in self.glycan_to_index.items()}
        self.simplify()

    def match(self, spectrum: DeconvolutedPeakSet, error_tolerance: float) -> DefaultDict[int, List[Tuple[SimpleFragment, float]]]:
        match_index = defaultdict(list)
        for fragment in self.fragments:
            peak = spectrum.has_peak(fragment.mass, error_tolerance)
            if peak is not None:
                for key in self.fragment_index[fragment.name]:
                    match_index[key].append((fragment, peak.index.neutral_mass))
        return match_index

    def simplify(self):
        id_to_frag_group = defaultdict(set)
        for f, members in self.fragment_index.items():
            for member in members:
                id_to_frag_group[member].add(f)

        groups = defaultdict(list)
        for member, group in id_to_frag_group.items():
            groups[frozenset(group)].append(member)

        counter = 0
        new_fragment_index = defaultdict(list)
        new_glycan_index = {}
        for frag_group, members in groups.items():
            new_id = counter
            counter += 1
            for frag in frag_group:
                new_fragment_index[frag].append(new_id)
            for member in members:
                new_glycan_index[self.index_to_glycan[member]] = new_id

        self.glycan_to_index = new_glycan_index
        self.fragment_index = new_fragment_index
        self.index_to_glycan = defaultdict(list)
        for k, v in self.glycan_to_index.items():
            self.index_to_glycan[v].append(k)


try:
    from glycresoft._c.tandem.oxonium_ions import OxoniumIndex, SignatureIonIndex, SignatureIonIndexMatch
except ImportError:
    SignatureIonIndex = object
    SignatureIonIndexMatch = object


@dataclass(frozen=True)
class OxoniumFilterState:
    scan_id: str
    g_score: float = field(hash=False)
    g_score_pass: bool
    oxonium_ions: FrozenSet[str]


@dataclass
class OxoniumFilterReport(MutableSequence[OxoniumFilterState]):
    records: List[OxoniumFilterState] = field(default_factory=list)

    def copy(self):
        return self.__class__(self.records.copy())

    def __getitem__(self, i):
        return self.records[i]

    def __setitem__(self, i, v):
        self.records[i] = v

    def __delitem__(self, i):
        del self.records[i]

    def append(self, value: OxoniumFilterState):
        self.records.append(value)

    def extend(self, values: Iterable[OxoniumFilterState]):
        self.records.extend(values)

    def insert(self, index: int, value: OxoniumFilterState):
        self.records.insert(index, value)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def prepare_report(self):
        passing: List[OxoniumFilterState] = []
        failing = []
        passing_scores=  []
        for rec in self:
            if rec.g_score_pass:
                passing.append(rec)
                passing_scores.append(rec.g_score)
            else:
                failing.append(rec)
        {
            "n_passing": len(passing),
            "n_failing": len(failing),
            "passing_distribution": passing_scores,
            "passing_median_score": median(passing_scores) if passing_scores else None
        }
