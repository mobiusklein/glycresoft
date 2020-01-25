# cython: embedsignature=True
from collections import defaultdict

cimport cython
from cpython cimport (
    PyList_Append, PyList_Size, PyList_GetItem,
    PyFloat_AsDouble, PyTuple_Size, PyTuple_GetItem,
    PyObject, PyDict_GetItem, PyDict_SetItem)

from libc cimport math

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet
from glycan_profiling._c.chromatogram_tree.mass_shift cimport MassShiftBase

np.import_array()

import warnings

from glycopeptidepy.structure.fragmentation_strategy import StubGlycopeptideStrategy, _AccumulatorBag
from glycan_profiling.chromatogram_tree import Unmodified as _Unmodified
from glycan_profiling.serialize import GlycanTypes

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
        public size_t size
        public size_t internal_size_approximation
        public size_t count
        public list glycan_types
        public dict _fragment_cache
        public Py_hash_t _hash

    cpdef bint is_n_glycan(self):
        return GlycanTypes_n_glycan in self.glycan_types

    cpdef bint is_o_glycan(self):
        return GlycanTypes_o_glycan in self.glycan_types

    cpdef bint is_gag_linker(self):
        return GlycanTypes_gag_linker in self.glycan_types

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
                shift['key'] = _AccumulatorBag(shift['key'])
                if shift["key"]['HexNAc'] <= 2 and shift["key"]["Hex"] <= 3:
                    shift['is_core'] = True
                else:
                    shift['is_core'] = False
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], shift['is_core']))
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
                shift['key'] = _AccumulatorBag(shift['key'])
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
                shift['key'] = _AccumulatorBag(shift['key'])
                fragment_structs.append(
                    CoarseStubGlycopeptideFragment(
                        shift['key'], shift['mass'], True))
            result = sorted(set(fragment_structs))
            PyDict_SetItem(self._fragment_cache, GlycanTypes_gag_linker, result)
            return result
        else:
            result = <list>ptemp
            return result



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
        n_fmatch = PyList_Size(self.fragment_matches)
        for i_fmatch in range(n_fmatch):
            fmatch = <CoarseStubGlycopeptideMatch>PyList_GetItem(self.fragment_matches, i_fmatch)
            fmass = fmatch.shift_mass
            n_peak = PyTuple_Size(fmatch.peaks_matched)
            for i_peak in range(n_peak):
                peak = <DeconvolutedPeak>PyTuple_GetItem(fmatch.peaks_matched, i_peak)
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

    def __init__(self, product_error_tolerance=1e-5, fragment_weight=0.56, core_weight=0.42):
        self.product_error_tolerance = product_error_tolerance
        self.fragment_weight = fragment_weight
        self.core_weight = core_weight

    cpdef CoarseGlycanMatch _match_fragments(self, object scan, double peptide_mass, list shifts, double mass_shift_tandem_mass=0.0):
        cdef:
            list fragment_matches
            double core_matched, core_theoretical, product_error_tolerance
            double n_frags_matched
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

        n = PyList_Size(shifts)

        for i in range(n):
            shift = <CoarseStubGlycopeptideFragment>PyList_GetItem(shifts, i)
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
        n = PyList_Size(matched_fragments)
        for i in range(n):
            matched_fragment = <CoarseStubGlycopeptideMatch>PyList_GetItem(matched_fragments, i)
            m = PyTuple_Size(matched_fragment.peaks_matched)
            for j in range(m):
                peak = <DeconvolutedPeak>PyTuple_GetItem(matched_fragment.peaks_matched, j)
                score += math.log(peak.intensity) * (1 - (
                    math.fabs(peak.neutral_mass - matched_fragment.mass) / matched_fragment.mass) ** 4) * coverage
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

@cython.boundscheck(False)
@cython.binding(True)
def GlycanFilteringPeptideMassEstimator_match(GlycanCoarseScorerBase self, scan, MassShiftBase mass_shift=Unmodified):
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
    intact_mass = scan.precursor_information.neutral_mass
    threshold_mass = (intact_mass + 1) - self.minimum_peptide_mass
    last_peptide_mass = 0
    glycan_combination_db = self.glycan_combination_db
    n = PyList_Size(glycan_combination_db)
    for i in range(n):
        glycan_combination = <GlycanCombinationRecordBase>PyList_GetItem(glycan_combination_db, i)
        # Stop searching when the peptide mass would be below the minimum peptide mass
        if threshold_mass < glycan_combination.dehydrated_mass:
            break
        peptide_mass = (
            intact_mass - glycan_combination.dehydrated_mass
        ) - mass_shift.mass
        if isclose(last_peptide_mass, peptide_mass):
            continue
        last_peptide_mass = peptide_mass
        best_score = 0
        best_match = None
        type_to_score = {}
        if glycan_combination.is_n_glycan():
            match_stat = self.n_glycan_coarse_score(scan, glycan_combination, mass_shift, peptide_mass)
            type_to_score[GlycanTypes_n_glycan] = match_stat
            score = PyFloat_AsDouble(<object>PyTuple_GetItem(match_stat, 0))
            if score > best_score:
                match = <CoarseGlycanMatch>PyTuple_GetItem(match_stat, 1)
                best_score = score
                best_match = match
        if glycan_combination.is_o_glycan():
            match_stat = self.o_glycan_coarse_score(scan, glycan_combination, mass_shift, peptide_mass)
            type_to_score[GlycanTypes_o_glycan] = match_stat
            score = PyFloat_AsDouble(<object>PyTuple_GetItem(match_stat, 0))
            if score > best_score:
                match = <CoarseGlycanMatch>PyTuple_GetItem(match_stat, 1)
                best_score = score
                best_match = match
        if glycan_combination.is_gag_linker():
            match_stat = self.gag_coarse_score(scan, glycan_combination, mass_shift, peptide_mass)
            type_to_score[GlycanTypes_gag_linker] = match_stat
            score = PyFloat_AsDouble(<object>PyTuple_GetItem(match_stat, 0))
            if score > best_score:
                match = <CoarseGlycanMatch>PyTuple_GetItem(match_stat, 1)
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
        result = GlycanMatchResult._create(
            peptide_mass,
            best_score, best_match, glycan_combination.size, type_to_score, recalibrated_peptide_mass)
        output.append(result)
    output = sorted(output, key=lambda x: x.score, reverse=1)
    return output