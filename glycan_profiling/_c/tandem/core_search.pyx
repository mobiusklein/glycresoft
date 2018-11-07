# cython: embedsignature=True
from collections import defaultdict

cimport cython
from cpython cimport PyList_Append, PyList_Size, PyList_GetItem, PyFloat_AsDouble, PyTuple_Size, PyTuple_GetItem

from libc cimport math

import numpy as np
cimport numpy as np

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

np.import_array()


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
        public tuple peaks_matched

    @staticmethod
    cdef CoarseStubGlycopeptideMatch _create(object key, double mass, tuple peaks_matched):
        cdef CoarseStubGlycopeptideMatch instance
        instance = CoarseStubGlycopeptideMatch.__new__(CoarseStubGlycopeptideMatch)
        instance.key = key
        instance.mass = mass
        instance.peaks_matched = peaks_matched
        return instance

    def __init__(self, key, mass, peaks_matched):
        self.key = key
        self.mass = mass
        self.peaks_matched = peaks_matched

    def __reduce__(self):
        return self.__class__, (self.key, self.mass, self.peaks_matched)

    def __repr__(self):
        return "%s(%s, %f, %r)" % (
            self.__class__.__name__,
            self.key, self.mass, self.peaks_matched
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
            DeconvolutedPeakSet peak_set
            size_t i, n
            bint has_tandem_shift
            tuple hits
            CoarseStubGlycopeptideFragment shift

        product_error_tolerance = (self.product_error_tolerance)
        fragment_matches = []

        core_matched = 0.0
        core_theoretical = 0.0
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
                fragment_matches.append(CoarseStubGlycopeptideMatch._create(shift.key, target_mass, hits))
            if has_tandem_shift:
                shifted_mass = target_mass + mass_shift_tandem_mass
                hits = peak_set.all_peaks_for(
                    shifted_mass, product_error_tolerance)
                if hits:
                    if is_core:
                        core_matched += 1
                    fragment_matches.append(CoarseStubGlycopeptideMatch._create(shift.key, shifted_mass, hits))

        return CoarseGlycanMatch._create(
            fragment_matches, float(len(fragment_matches)), float(len(shifts)), core_matched, core_theoretical)


    cpdef double _calculate_score(self, list matched_fragments, double n_matched, double n_theoretical,
                                  double core_matched, double core_theoretical):
        cdef:
            size_t i, j, n, m
            CoarseStubGlycopeptideMatch matched_fragment
            double score, ratio_fragments, ratio_core, coverage
            DeconvolutedPeak peak
            tuple matches
        ratio_fragments = (n_matched / n_theoretical)
        ratio_core = core_matched / core_theoretical
        coverage = (ratio_fragments ** (self.fragment_weight)) * (ratio_core ** (self.core_weight))
        score = 0
        n = PyList_Size(matched_fragments)
        for i in range(n):
            matched_fragment = <CoarseStubGlycopeptideMatch>PyList_GetItem(matched_fragments, i)
            m = PyTuple_Size(matched_fragment.peaks_matched)
            for j in range(m):
                peak = <DeconvolutedPeak>PyTuple_GetItem(matched_fragment.peaks_matched, j)
                score += math.log(peak.intensity) * (1 - (
                    math.fabs(peak.neutral_mass - matched_fragment.mass) / matched_fragment.mass) ** 4) * coverage
        return score