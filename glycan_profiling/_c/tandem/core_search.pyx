# cython: embedsignature=True
from collections import defaultdict

cimport cython
from cpython cimport PyList_Append, PyList_Size, PyList_GetItem

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


@cython.binding
def _n_glycan_match_stubs(object self, object scan, double peptide_mass, object glycan_combination, double mass_shift_tandem_mass=0.0):
        cdef:
            list shifts, fragment_matches
            double core_matched, core_theoretical, product_error_tolerance
            DeconvolutedPeakSet peak_set
            size_t i, j, k, n, m
            bint has_tandem_shift
            tuple hits
            CoarseStubGlycopeptideFragment shift

        product_error_tolerance = self.product_error_tolerance
        shifts = glycan_combination.get_n_glycan_fragments()
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

        return fragment_matches, float(len(fragment_matches)), float(len(shifts)), core_matched, core_theoretical