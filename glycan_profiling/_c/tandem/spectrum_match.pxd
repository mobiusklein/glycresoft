cimport cython

from ms_deisotope._c.peak_set cimport DeconvolutedPeak
from glycan_profiling._c.chromatogram_tree.mass_shift cimport MassShiftBase


cdef class SpectrumMatchBase(object):
    cdef:
        public object scan
        public object target
        public MassShiftBase mass_shift


cdef class ScoreSet(object):
    cdef:
        public float glycopeptide_score
        public float peptide_score
        public float glycan_score
        public float glycan_coverage
        public float stub_glycopeptide_intensity_utilization
        public float oxonium_ion_intensity_utilization
        public int n_stub_glycopeptide_matches

    cpdef bint _lt(self, ScoreSet other)
    cpdef bint _gt(self, ScoreSet other)
    cpdef bint _eq(self, ScoreSet other)
    cpdef bytearray pack(self)

    @staticmethod
    cdef ScoreSet _create(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                          float stub_glycopeptide_intensity_utilization, float oxonium_ion_intensity_utilization,
                          int n_stub_glycopeptide_matches)

cdef class FDRSet(object):
    cdef:
        public double total_q_value
        public double peptide_q_value
        public double glycan_q_value
        public double glycopeptide_q_value

    cpdef bint _lt(self, FDRSet other)
    cpdef bint _gt(self, FDRSet other)
    cpdef bint _eq(self, FDRSet other)
    cpdef bytearray pack(self)

    @staticmethod
    cdef FDRSet _create(double total_q_value, double peptide_q_value, double glycan_q_value, double glycopeptide_q_value)


@cython.final
@cython.freelist(1000)
cdef class PeakFoundRecord(object):
    cdef:
        public DeconvolutedPeak peak
        public bint checked

    @staticmethod
    cdef inline PeakFoundRecord _create(DeconvolutedPeak peak, bint checked):
        cdef PeakFoundRecord self = PeakFoundRecord.__new__(PeakFoundRecord)
        self.peak = peak
        self.checked = checked
        return self


@cython.final
cdef class PeakLabelMap(object):
    cdef:
        public dict name_to_peaks

    @staticmethod
    cdef PeakLabelMap _create()

    cdef inline void add(self, str name, DeconvolutedPeak peak)
    cdef inline PeakFoundRecord get(self, str name)