# cython: embedsignature=True
from collections import defaultdict

cimport cython

from cpython.object cimport PyObject
from cpython.float cimport PyFloat_AsDouble
from cpython.int cimport PyInt_AsLong, PyInt_FromLong
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM, PyList_Append, PyList_Size, PyList_GetItem
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_Size, PyTuple_GetItem
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_Items, PyDict_Values
from cpython.set cimport PySet_Add, PySet_GET_SIZE

from libc cimport math

import numpy as np
cimport numpy as np
from numpy cimport npy_uint16 as uint16_t, npy_int32 as int32_t

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet
from glycresoft._c.chromatogram_tree.mass_shift cimport MassShiftBase

np.import_array()

import warnings

from glypy.composition.ccomposition cimport CComposition
from glycopeptidepy._c.structure.fragmentation_strategy.glycan cimport StubGlycopeptideStrategy, GlycanCompositionFragment
from glycopeptidepy._c.count_table cimport CountTable
from glycresoft.chromatogram_tree import Unmodified as _Unmodified
from glycresoft.serialize import GlycanTypes

from glypy.structure.glycan_composition import HashableGlycanComposition as _HashableGlycanComposition

cdef object HashableGlycanComposition = _HashableGlycanComposition
cdef MassShiftBase Unmodified = _Unmodified
cdef object GlycanTypes_n_glycan, GlycanTypes_o_glycan, GlycanTypes_gag_linker


GlycanTypes_n_glycan = GlycanTypes.n_glycan
GlycanTypes_o_glycan = GlycanTypes.o_glycan
GlycanTypes_gag_linker = GlycanTypes.gag_linker


cdef class GlycanCombinationRecordBase(object):
    cdef:
        public size_t id
        public double dehydrated_mass
        public object composition
        public uint16_t size
        public uint16_t internal_size_approximation
        public uint16_t side_group_count
        public uint16_t count
        public list glycan_types
        public dict _fragment_cache
        public Py_hash_t _hash
        public dict fragment_set_properties

    cpdef bint is_n_glycan(self):
        return GlycanTypes_n_glycan in self.glycan_types

    cpdef bint is_o_glycan(self):
        return GlycanTypes_o_glycan in self.glycan_types

    cpdef bint is_gag_linker(self):
        return GlycanTypes_gag_linker in self.glycan_types

    @cython.cdivision(True)
    cpdef double normalized_size(self):
        cdef:
            double k
            double d
            uint16_t n
        n = self.internal_size_approximation
        if n == 0:
            return 0.0
        if self.side_group_count > 0:
            k = 1.0
        else:
            k = 2.0
        d = max(n * math.log(n) / k, n)
        return d

    cpdef list get_n_glycan_fragments(self):
        cdef:
            PyObject* ptemp
            list result, fragment_structs
        ptemp = PyDict_GetItem(self._fragment_cache, GlycanTypes_n_glycan)
        if ptemp == NULL:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.n_glycan_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                if shift["key"]['HexNAc'] <= 2 and shift["key"]["Hex"] <= 3:
                    is_core = True
                else:
                    is_core = False
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], is_core))
            result = sorted(set(fragment_structs))
            PyDict_SetItem(self._fragment_cache, GlycanTypes_n_glycan, result)
            return result
        else:
            result = <list>ptemp
            return result

    cpdef list get_o_glycan_fragments(self):
        cdef:
            PyObject* ptemp
            list result, fragment_structs
        ptemp = PyDict_GetItem(self._fragment_cache, GlycanTypes_o_glycan)
        if ptemp == NULL:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.o_glycan_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            result = sorted(set(fragment_structs))
            PyDict_SetItem(self._fragment_cache, GlycanTypes_o_glycan, result)
            return result
        else:
            result = <list>ptemp
            return result

    cpdef list get_gag_linker_glycan_fragments(self):
        cdef:
            PyObject* ptemp
            list result, fragment_structs
        ptemp = PyDict_GetItem(self._fragment_cache, GlycanTypes_gag_linker)
        if ptemp == NULL:
            strategy = StubGlycopeptideStrategy(None, extended=True)
            shifts = strategy.gag_linker_composition_fragments(self.composition, 1, 0)
            fragment_structs = []
            for shift in shifts:
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            result = sorted(set(fragment_structs))
            PyDict_SetItem(self._fragment_cache, GlycanTypes_gag_linker, result)
            return result
        else:
            result = <list>ptemp
            return result

    cpdef clear(self):
        self._fragment_cache.clear()


cdef class CoarseStubGlycopeptideFragment(object):
    cdef:
        public object key
        public double mass
        public bint is_core

    def __init__(self, key, mass, is_core):
        self.key = key
        self.mass = mass
        self.is_core = is_core

    def __reduce__(self):
        return self.__class__, (self.key, self.mass, self.is_core)

    def __repr__(self):
        return "%s(%s, %f, %r)" % (
            self.__class__.__name__,
            self.key, self.mass, self.is_core
        )

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


@cython.freelist(100000)
@cython.final
cdef class CoarseStubGlycopeptideMatch(object):
    cdef:
        public object key
        public double mass
        public double shift_mass
        public tuple peaks_matched

    @staticmethod
    cdef CoarseStubGlycopeptideMatch _create(object key, double mass, double shift_mass, tuple peaks_matched):
        cdef CoarseStubGlycopeptideMatch instance
        instance = CoarseStubGlycopeptideMatch.__new__(CoarseStubGlycopeptideMatch)
        instance.key = key
        instance.mass = mass
        instance.shift_mass = shift_mass
        instance.peaks_matched = peaks_matched
        return instance

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


@cython.freelist(100000)
@cython.final
cdef class CoarseGlycanMatch(object):
    cdef:
        public list fragment_matches
        public double n_matched
        public double n_theoretical
        public double core_matched
        public double core_theoretical

    @staticmethod
    cdef CoarseGlycanMatch _create(list fragment_matches, double n_matched, double n_theoretical, double core_matched, double core_theoretical):
        cdef CoarseGlycanMatch inst = CoarseGlycanMatch.__new__(CoarseGlycanMatch)
        inst.fragment_matches = fragment_matches
        inst.n_matched = n_matched
        inst.n_theoretical = n_theoretical
        inst.core_matched = core_matched
        inst.core_theoretical = core_theoretical
        return inst

    def __init__(self, fragment_matches, n_matched, n_theoretical, core_matched, core_theoretical):
        self.fragment_matches = list(fragment_matches)
        self.n_matched = n_matched
        self.n_theoretical = n_theoretical
        self.core_matched = core_matched
        self.core_theoretical = core_theoretical

    def __iter__(self):
        yield self.fragment_matches
        yield self.n_matched
        yield self.n_theoretical
        yield self.core_matched
        yield self.core_theoretical

    cpdef double estimate_peptide_mass(self):
        cdef:
            double weighted_mass_acc
            double weight_acc
            double fmass

            size_t i_fmatch, n_fmatch, i_peak, n_peak
            CoarseStubGlycopeptideMatch fmatch
            DeconvolutedPeak peak
        weighted_mass_acc = 0
        weight_acc = 0
        n_fmatch = PyList_GET_SIZE(self.fragment_matches)
        for i_fmatch in range(n_fmatch):
            fmatch = <CoarseStubGlycopeptideMatch>PyList_GET_ITEM(self.fragment_matches, i_fmatch)
            fmass = fmatch.shift_mass
            n_peak = PyTuple_GET_SIZE(fmatch.peaks_matched)
            for i_peak in range(n_peak):
                peak = <DeconvolutedPeak>PyTuple_GET_ITEM(fmatch.peaks_matched, i_peak)
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


cdef class GlycanCoarseScorerBase(object):
    cdef:
        public double product_error_tolerance
        public double fragment_weight
        public double core_weight
        public bint recalibrate_peptide_mass

    def __init__(self, product_error_tolerance=1e-5, fragment_weight=0.56, core_weight=0.42, recalibrate_peptide_mass=False):
        self.product_error_tolerance = product_error_tolerance
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight
        self.recalibrate_peptide_mass = recalibrate_peptide_mass

    @cython.boundscheck(False)
    cpdef CoarseGlycanMatch _match_fragments(self, object scan, double peptide_mass, list shifts, double mass_shift_tandem_mass=0.0):
        cdef:
            list fragment_matches
            double core_matched, core_theoretical, product_error_tolerance
            double n_frags_matched, approximate_size_normalizer
            DeconvolutedPeakSet peak_set
            size_t i, n
            bint has_tandem_shift
            tuple hits
            CoarseStubGlycopeptideFragment shift

        product_error_tolerance = (self.product_error_tolerance)
        fragment_matches = []

        core_matched = 0.0
        core_theoretical = 0.0
        n_frags_matched = 0.0
        has_tandem_shift = abs(mass_shift_tandem_mass) > 0
        peak_set = scan.deconvoluted_peak_set

        n = PyList_GET_SIZE(shifts)

        for i in range(n):
            shift = <CoarseStubGlycopeptideFragment>PyList_GET_ITEM(shifts, i)
            if shift.is_core:
                is_core = True
                core_theoretical += 1
            target_mass = shift.mass + peptide_mass
            hits = peak_set.all_peaks_for(target_mass, product_error_tolerance)
            if hits:
                if is_core:
                    core_matched += 1
                fragment_matches.append(
                    CoarseStubGlycopeptideMatch._create(shift.key, target_mass, shift.mass, hits))
                n_frags_matched += 1
            if has_tandem_shift:
                shifted_mass = target_mass + mass_shift_tandem_mass
                hits = peak_set.all_peaks_for(
                    shifted_mass, product_error_tolerance)
                if hits:
                    if is_core:
                        core_matched += 1
                    fragment_matches.append(
                        CoarseStubGlycopeptideMatch._create(
                            shift.key, shifted_mass, shift.mass + mass_shift_tandem_mass, hits))
                    n_frags_matched += 1

        return CoarseGlycanMatch._create(
            fragment_matches, n_frags_matched, float(n), core_matched, core_theoretical)

    @cython.cdivision(True)
    cpdef double _calculate_score(self, CoarseGlycanMatch glycan_match):
        cdef:
            size_t i, j, n, m
            CoarseStubGlycopeptideMatch matched_fragment
            double score, ratio_fragments, ratio_core, coverage
            DeconvolutedPeak peak
            tuple matches
            list matched_fragments

        ratio_fragments = (<double>glycan_match.n_matched) / glycan_match.n_theoretical
        ratio_core = <double>glycan_match.core_matched / glycan_match.core_theoretical
        coverage = (ratio_fragments ** (self.fragment_weight)) * (ratio_core ** (self.core_weight))
        score = 0
        matched_fragments = glycan_match.fragment_matches
        n = PyList_GET_SIZE(matched_fragments)
        for i in range(n):
            matched_fragment = <CoarseStubGlycopeptideMatch>PyList_GET_ITEM(matched_fragments, i)
            m = PyTuple_GET_SIZE(matched_fragment.peaks_matched)
            for j in range(m):
                peak = <DeconvolutedPeak>PyTuple_GET_ITEM(matched_fragment.peaks_matched, j)
                score += math.log(peak.intensity) * (1 - (
                    math.fabs(peak.neutral_mass - matched_fragment.mass) / matched_fragment.mass) ** 4)
        score *= coverage
        return score

    cpdef CoarseGlycanMatch _n_glycan_match_stubs(self, scan, double peptide_mass,
                                                  GlycanCombinationRecordBase glycan_combination,
                                                  double mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_n_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    cpdef CoarseGlycanMatch _o_glycan_match_stubs(self, scan, double peptide_mass,
                                                  GlycanCombinationRecordBase glycan_combination,
                                                  double mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_o_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    cpdef CoarseGlycanMatch _gag_match_stubs(self, scan, double peptide_mass,
                                             GlycanCombinationRecordBase glycan_combination,
                                             double mass_shift_tandem_mass=0.0):
        shifts = glycan_combination.get_gag_linker_glycan_fragments()
        return self._match_fragments(scan, peptide_mass, shifts, mass_shift_tandem_mass)

    cdef CoarseGlycanMatch _get_n_glycan_coarse_score(self, object scan,
                                                      GlycanCombinationRecordBase glycan_combination,
                                                      MassShiftBase mass_shift, double peptide_mass,
                                                      double* scoreout):
        if peptide_mass < 0:
            scoreout[0] = -1e6
            return None
        glycan_match = self._n_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        scoreout[0] = score
        return glycan_match

    cdef CoarseGlycanMatch _get_o_glycan_coarse_score(self, object scan,
                                                      GlycanCombinationRecordBase glycan_combination,
                                                      MassShiftBase mass_shift, double peptide_mass,
                                                      double* scoreout):
        if peptide_mass < 0:
            scoreout[0] = -1e6
            return None
        glycan_match = self._o_glycan_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        scoreout[0] = score
        return glycan_match

    cdef CoarseGlycanMatch _get_gag_coarse_score(self, object scan,
                                                 GlycanCombinationRecordBase glycan_combination,
                                                 MassShiftBase mass_shift, double peptide_mass,
                                                 double* scoreout):
        if peptide_mass < 0:
            scoreout[0] = -1e6
            return None
        glycan_match = self._gag_match_stubs(
            scan, peptide_mass, glycan_combination, mass_shift_tandem_mass=mass_shift.tandem_mass)
        score = self._calculate_score(glycan_match)
        scoreout[0] = score
        return glycan_match


cdef bint isclose(double a, double b, double rtol=1e-05, double atol=1e-08):
    return abs(a - b) <= atol + rtol * abs(b)


@cython.freelist(1000)
cdef class GlycanMatchResult(object):
    cdef:
        public double peptide_mass
        public double score
        public CoarseGlycanMatch match
        public size_t glycan_size
        public dict glycan_types
        public double recalibrated_peptide_mass


    @staticmethod
    cdef GlycanMatchResult _create(double peptide_mass, double score, CoarseGlycanMatch match, size_t glycan_size,
                                   dict glycan_types, double recalibrated_peptide_mass):
        cdef:
            GlycanMatchResult self

        self = GlycanMatchResult.__new__(GlycanMatchResult)
        self.peptide_mass = peptide_mass
        self.score = score
        self.match = match
        self.glycan_size = glycan_size
        self.glycan_types = glycan_types
        self.recalibrated_peptide_mass = recalibrated_peptide_mass
        return self

    def __init__(self, peptide_mass, score, match, glycan_size, glycan_types, recalibrated_peptide_mass):
        self.peptide_mass = peptide_mass
        self.score = score
        self.match = match
        self.glycan_size = glycan_size
        self.glycan_types = glycan_types
        self.recalibrated_peptide_mass = recalibrated_peptide_mass

    @property
    def fragment_match_count(self):
        match = self.match
        if match is None:
            return 0
        return match.n_matched

    def __repr__(self):
        template = (
            "{self.__class__.__name__}({self.peptide_mass}, {self.score},"
            " {self.match}, {self.glycan_size}, {self.glycan_types}, {self.recalibrated_peptide_mass})")
        return template.format(self=self)


@cython.freelist(1000)
@cython.final
cdef class ScoreCoarseGlycanMatchPair(object):
    cdef:
        public double score
        public CoarseGlycanMatch match

    @staticmethod
    cdef ScoreCoarseGlycanMatchPair _create(double score, CoarseGlycanMatch match):
        cdef ScoreCoarseGlycanMatchPair self = ScoreCoarseGlycanMatchPair.__new__(ScoreCoarseGlycanMatchPair)
        self.score = score
        self.match = match
        return self

    def __init__(self, score, match):
        self.score = score
        self.match = match

    def __getitem__(self, i):
        if i == 0:
            return self.score
        elif i == 1:
            return self.match
        else:
            raise IndexError(i)

    def __len__(self):
        return 2

    def __repr__(self):
        return "(%f, %r)" % (self.score, self.match)


cpdef double score_getter(object x):
    return (<GlycanMatchResult>x).score


@cython.boundscheck(False)
@cython.binding(True)
cpdef GlycanFilteringPeptideMassEstimator_match(GlycanCoarseScorerBase self, scan, MassShiftBase mass_shift=Unmodified, query_mass=None):
    cdef:
        list output, glycan_combination_db
        double intact_mass, threshold_mass, last_peptide_mass, peptide_mass
        GlycanCombinationRecordBase glycan_combination
        dict type_to_score
        double best_score, score
        size_t i, n
        tuple match_stat
        CoarseGlycanMatch best_match, match

    output = []
    if query_mass is None:
        intact_mass = scan.precursor_information.neutral_mass
    else:
        intact_mass = query_mass
    threshold_mass = (intact_mass + 1) - self.minimum_peptide_mass
    last_peptide_mass = 0
    glycan_combination_db = self.glycan_combination_db
    n = PyList_Size(glycan_combination_db)
    for i in range(n):
        glycan_combination = <GlycanCombinationRecordBase>PyList_GET_ITEM(glycan_combination_db, i)
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
            score = 0
            match = self._get_n_glycan_coarse_score(scan, glycan_combination, mass_shift, peptide_mass, &score)
            type_to_score[GlycanTypes_n_glycan] = ScoreCoarseGlycanMatchPair._create(score, match)
            if score > best_score:
                best_score = score
                best_match = match
        if glycan_combination.is_o_glycan():
            score = 0
            match = self._get_o_glycan_coarse_score(scan, glycan_combination, mass_shift, peptide_mass, &score)
            type_to_score[GlycanTypes_o_glycan] = ScoreCoarseGlycanMatchPair._create(score, match)
            if score > best_score:
                best_score = score
                best_match = match
        if glycan_combination.is_gag_linker():
            match = self._get_gag_coarse_score(scan, glycan_combination, mass_shift, peptide_mass, &score)
            type_to_score[GlycanTypes_gag_linker] = ScoreCoarseGlycanMatchPair._create(score, match)
            if score > best_score:
                best_score = score
                best_match = match

        if best_match is not None and self.recalibrate_peptide_mass:
            recalibrated_peptide_mass = best_match.estimate_peptide_mass()
            if recalibrated_peptide_mass > 0:
                if abs(recalibrated_peptide_mass - peptide_mass) > 0.5:
                    warnings.warn("Re-estimated peptide mass error is large: %f vs %f" % (
                        peptide_mass, recalibrated_peptide_mass))
        else:
            recalibrated_peptide_mass = 0
        result = GlycanMatchResult._create(
            peptide_mass,
            best_score, best_match, glycan_combination.size, type_to_score, recalibrated_peptide_mass)
        output.append(result)
    output = sorted(output, key=score_getter, reverse=1)
    return output


cdef class IndexGlycanCompositionFragment(GlycanCompositionFragment):
    cdef:
        public int index
        public str name

    def __init__(self, mass, composition, key, is_extended=False):
        self.mass = mass
        self.composition = composition
        self.key = key
        self.is_extended = is_extended
        self._hash_key = -1
        self.index = -1
        self.name = None

    @staticmethod
    cdef IndexGlycanCompositionFragment _create(double mass, CComposition composition, CountTable key, bint is_extended):
        cdef:
            IndexGlycanCompositionFragment self
        self = IndexGlycanCompositionFragment.__new__(IndexGlycanCompositionFragment)
        self.mass = mass
        self.composition = composition
        self.key = key
        self.is_extended = is_extended
        self._hash_key = -1
        self.index = -1
        self.name = None
        return self

    cpdef GlycanCompositionFragment copy(self):
        cdef:
            IndexGlycanCompositionFragment inst
        inst = IndexGlycanCompositionFragment._create(
            self.mass,
            self.composition.clone() if self.composition is not None else None, self.key.copy(),
            self.is_extended)
        inst._hash_key = self._hash_key
        inst.index = self.index
        inst.name = self.name
        return inst

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


@cython.freelist(1000)
cdef class ComplementFragment(object):
    cdef:
        public double mass
        public list keys

    def __init__(self, mass, keys=None):
        self.mass = mass
        self.keys = keys or []

    def __repr__(self):
        return "{self.__class__.__name__}({self.mass})".format(self=self)


@cython.freelist(10000)
cdef class PartialGlycanSolution(object):
    cdef:
        public double peptide_mass
        public double score
        public set core_matches
        public set fragment_matches
        public int32_t glycan_index

    def __init__(self, peptide_mass=-1, score=0, core_matches=None, fragment_matches=None, int32_t glycan_index=-1):
        if core_matches is None:
            core_matches = set()
        if fragment_matches is None:
            fragment_matches = set()
        self.peptide_mass = peptide_mass
        self.score = score
        self.core_matches = core_matches
        self.fragment_matches = fragment_matches
        self.glycan_index = glycan_index

    @staticmethod
    cdef PartialGlycanSolution _create():
        cdef PartialGlycanSolution self = PartialGlycanSolution.__new__(PartialGlycanSolution)
        self.peptide_mass = -1
        self.score = 0.0
        self.core_matches = set()
        self.fragment_matches = set()
        self.glycan_index = -1
        return self

    def __repr__(self):
        template =("{self.__class__.__name__}({self.peptide_mass}, {self.score}, "
                   "{self.core_matches}, {self.fragment_matches}, {self.glycan_index})")
        return template.format(self=self)

    @property
    def fragment_match_count(self):
        return len(self.fragment_matches)


cdef class GlycanFragmentIndex(object):
    '''A fast and sparse in-memory fragment ion index for quickly matching peaks against multiple
    glycan composition complements.

    Based upon the complement ion indexing strategy described in [1]_.

    Attributes
    ----------
    members : :class:`list` of :class:`GlycanCombinationRecord`
        The glycan composition combinations in this index.
    unique_fragments : :class:`dict` of :class:`str` -> :class:`dict` of :class:`str` -> :class:`IndexGlycanCompositionFragment`
        An internment table for each glycosylation type mapping fragment name to glycan composition fragments
    fragment_index : :class:`dict` of :class:`int` -> :class:`list` of :class:`ComplementFragment`
        A sparse index mapping :attr:`resolution` scaled masses to bins. The mass values and binned fragments are complements of
        true fragments.
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

    cdef:
        public list members
        public object glycosylation_type
        public dict unique_fragments
        public list _fragments
        public dict fragment_index
        public long counter
        public double fragment_weight
        public double core_weight
        public double lower_bound
        public double upper_bound
        public double resolution

    def __init__(self, members=None, fragment_weight=0.56, core_weight=0.42, resolution=100.0):
        self.members = members or []
        self.unique_fragments = dict()
        self._fragments = []
        self.fragment_index = dict()
        self.counter = 0
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight
        self.lower_bound = float('inf')
        self.upper_bound = 0
        self.resolution = resolution

    cpdef IndexGlycanCompositionFragment _intern(self, fragment, glycosylation_type):
        key = str(HashableGlycanComposition(fragment.key))
        try:
            cache_for_type = self.unique_fragments[glycosylation_type]
        except KeyError:
            cache_for_type = self.unique_fragments[glycosylation_type] = dict()
        try:
            return cache_for_type[key]
        except KeyError:
            fragment = IndexGlycanCompositionFragment._create(
                fragment.mass, None, fragment.key, not fragment.is_core)
            cache_for_type[key] = fragment
            fragment.name = key
            fragment.index = self.counter
            self._fragments.append(fragment)
            self.counter += 1
            return fragment

    cpdef long query_width(self, double mass, double error_tolerance):
        return <long>(mass * error_tolerance * self.resolution) + 1

    cpdef long bin_index(self, double mass):
        return <long>(mass * self.resolution)

    def build(self):
        cdef:
            object key
            long n_frags, n_core, j
            double mass, d, low, high
            PyObject* ptemp
            list comp_frags, frags
            size_t frags_i, frags_n, comp_i, comp_n
            IndexGlycanCompositionFragment frag
            ComplementFragment comp
            GlycanCombinationRecordBase member

        self.members.sort(key=lambda x: (x.dehydrated_mass, x.id))
        self.fragment_index.clear()
        j = 0
        low = float('inf')
        high = 0
        for i, member in enumerate(self.members):
            mass = member.dehydrated_mass
            for glycosylation_type in member.glycan_types:
                n_core = 0
                n_frags = 0
                frags = self._get_fragments(member, glycosylation_type)
                for frag_i in range(PyList_GET_SIZE(frags)):
                    frag = self._intern(<object>PyList_GET_ITEM(frags, frag_i), glycosylation_type)
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
                    key = PyInt_FromLong(self.bin_index(d))
                    ptemp = PyDict_GetItem(self.fragment_index, key)
                    if ptemp == NULL:
                        comp_frags = []
                        PyDict_SetItem(self.fragment_index, key, comp_frags)
                    else:
                        comp_frags = <list>ptemp
                    for comp_i in range(PyList_GET_SIZE(comp_frags)):
                        comp = <ComplementFragment>PyList_GET_ITEM(comp_frags, comp_i)
                        if abs(comp.mass - d) <= 1e-3:
                            PyList_Append(comp.keys, (frag, i, glycosylation_type))
                            break
                    else:
                        comp_frags.append(ComplementFragment(d, [(frag, i, glycosylation_type)]))
                        j += 1
                member.fragment_set_properties[glycosylation_type] = (n_core, n_frags)
            member.clear()
        self.lower_bound = low
        self.upper_bound = high
        return j

    cpdef list _get_fragments(self, GlycanCombinationRecordBase gcr, object glycosylation_type):
        if glycosylation_type == GlycanTypes_n_glycan:
            return gcr.get_n_glycan_fragments()
        elif glycosylation_type == GlycanTypes_o_glycan:
            return gcr.get_o_glycan_fragments()
        elif glycosylation_type == GlycanTypes_gag_linker:
            return gcr.get_gag_linker_glycan_fragments()
        else:
            raise ValueError(glycosylation_type)

    @cython.cdivision
    cdef int _match_fragments(self, double delta_mass, DeconvolutedPeak peak, double error_tolerance, dict result):
            cdef:
                object idx, glycosylation_type
                tuple frag_id
                list comp_frags, keys
                dict sols_for_glycan
                long key, width, off
                ComplementFragment comp_frag
                IndexGlycanCompositionFragment frag
                GlycanCombinationRecordBase rec
                PartialGlycanSolution sol
                size_t comp_i, comp_n, key_i, key_n, n_core, n_frag
                PyObject* ptemp

            key = self.bin_index(delta_mass)
            width = <long>(peak.neutral_mass * error_tolerance * self.resolution) + 1

            for off in range(-width, width + 1):
                ptemp = PyDict_GetItem(self.fragment_index, key + off)
                if ptemp == NULL:
                    continue
                else:
                    comp_frags = <list>ptemp
                comp_n = PyList_GET_SIZE(comp_frags)
                for comp_i in range(comp_n):
                    comp_frag = <ComplementFragment>PyList_GET_ITEM(comp_frags, comp_i)
                    if abs(comp_frag.mass - delta_mass) / peak.neutral_mass <= error_tolerance:
                        keys = comp_frag.keys
                        key_n = PyList_GET_SIZE(keys)
                        for key_i in range(key_n):
                            frag_id = <tuple>PyList_GET_ITEM(keys, key_i)
                            frag = <IndexGlycanCompositionFragment>PyTuple_GET_ITEM(frag_id, 0)
                            idx = <object>PyTuple_GET_ITEM(frag_id, 1)
                            glycosylation_type = <object>PyTuple_GET_ITEM(frag_id, 2)

                            # Fetch the dict for the glycan's solutions across glycosylation types.
                            # If it doesn't exist, populate it now.
                            ptemp = PyDict_GetItem(result, idx)
                            if ptemp == NULL:
                                sols_for_glycan = dict()
                                PyDict_SetItem(result, idx, sols_for_glycan)
                            else:
                                sols_for_glycan = <dict>ptemp
                            # From solutions for the current glycan across glycosylation types,
                            # fetch the solution for the matched glycosylation type. If it does not
                            # exist, populate it now.
                            ptemp = PyDict_GetItem(sols_for_glycan, glycosylation_type)
                            if ptemp == NULL:
                                sol = PartialGlycanSolution._create()
                                sol.glycan_index = idx
                                PyDict_SetItem(sols_for_glycan, glycosylation_type, sol)
                            else:
                                sol = <PartialGlycanSolution>ptemp
                            sol.score += math.log10(peak.intensity) * (
                                1 - ((abs(delta_mass - comp_frag.mass) / peak.neutral_mass) / error_tolerance) ** 4)
                            if not frag.is_extended:
                                PySet_Add(sol.core_matches, frag.index)
                            PySet_Add(sol.fragment_matches, frag.index)
            return 0

    @cython.cdivision
    cpdef list match(self, scan, double error_tolerance=1e-5, MassShiftBase mass_shift=Unmodified, query_mass=None):
        cdef:
            double precursor_mass, d
            double best_score
            PartialGlycanSolution sol, best_solution
            dict result, sols_for_glycan
            DeconvolutedPeakSet peak_set
            DeconvolutedPeak peak
            object glycosylation_type
            tuple id_sol, glytype_sol
            list out, solutions
            double coverage, mass_shift_tandem_mass, mass_shift_mass
            GlycanCombinationRecordBase rec
            size_t peak_i, n_peaks, n_core, n_frag, sols_i, sols_n
            PyObject* ptemp

        if query_mass is None:
            precursor_mass = scan.precursor_information.neutral_mass
        else:
            precursor_mass = query_mass

        mass_shift_mass = mass_shift.mass
        mass_shift_tandem_mass = mass_shift.tandem_mass

        peak_set = scan.deconvoluted_peak_set
        n_peaks = peak_set.get_size()
        result = dict()
        for peak_i in range(n_peaks):
            peak = peak_set.getitem(peak_i)
            # Subtract the precursor mass shift mass to match unmodified peaks
            d = (precursor_mass - peak.neutral_mass - mass_shift_mass)
            self._match_fragments(d, peak, error_tolerance, result)
            if mass_shift_tandem_mass != 0:
                # Add back the tandem mass shift to match modified peaks
                d = (precursor_mass - peak.neutral_mass - mass_shift_mass + mass_shift_tandem_mass)
                self._match_fragments(d, peak, error_tolerance, result)

        out = []
        keys = PyDict_Items(result)
        key_n = PyList_GET_SIZE(keys)
        for key_i in range(key_n):
            id_sol = <tuple>PyList_GET_ITEM(keys, key_i)
            idx = <object>PyTuple_GET_ITEM(id_sol, 0)
            sols_for_glycan = <dict>PyTuple_GET_ITEM(id_sol, 1)
            rec = <GlycanCombinationRecordBase>PyList_GET_ITEM(self.members, idx)

            best_score = -1e6
            best_solution = None
            solutions = PyDict_Items(sols_for_glycan)
            sols_n = PyList_GET_SIZE(solutions)
            for sols_i in range(sols_n):
                glytype_sol = <tuple>PyList_GET_ITEM(solutions, sols_i)

                glycosylation_type = <object>PyTuple_GET_ITEM(glytype_sol, 0)
                sol = <PartialGlycanSolution>PyTuple_GET_ITEM(glytype_sol, 1)
                # Reusing container variable lazily
                id_sol = <tuple>PyDict_GetItem(rec.fragment_set_properties, glycosylation_type)
                n_core = PyInt_AsLong(<object>PyTuple_GET_ITEM(id_sol, 0))
                n_frag = PyInt_AsLong(<object>PyTuple_GET_ITEM(id_sol, 1))
                coverage = (PySet_GET_SIZE(sol.core_matches) * 1.0 / n_core) ** self.core_weight * (
                    PySet_GET_SIZE(sol.fragment_matches) * 1.0 / n_frag) ** self.fragment_weight
                sol.score *= coverage
                sol.peptide_mass = precursor_mass - rec.dehydrated_mass - mass_shift_mass
                if best_score < sol.score and sol.peptide_mass > 0:
                    best_score = sol.score
                    best_solution = sol
            if best_solution is not None:
                out.append(best_solution)

        out.sort(key=partial_score_getter, reverse=True)
        return out


cpdef double partial_score_getter(object x):
    return (<PartialGlycanSolution>x).score