cimport cython


cdef class ScoreSet(object):
    cdef:
        public float glycopeptide_score
        public float peptide_score
        public float glycan_score
        public float glycan_coverage
        public float stub_glycopeptide_intensity_utilization
        public int n_stub_glycopeptide_matches

    cpdef bint _lt(self, ScoreSet other)
    cpdef bint _gt(self, ScoreSet other)
    cpdef bint _eq(self, ScoreSet other)
    cpdef bytearray pack(self)

    @staticmethod
    cdef ScoreSet _create(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                          float stub_glycopeptide_intensity_utilization, int n_stub_glycopeptide_matches)

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