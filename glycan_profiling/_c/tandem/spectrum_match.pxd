cimport cython


cdef class ScoreSet(object):
    cdef:
        public double glycopeptide_score
        public double peptide_score
        public double glycan_score
        public double glycan_coverage

    cpdef bint _lt(self, ScoreSet other)
    cpdef bint _gt(self, ScoreSet other)
    cpdef bint _eq(self, ScoreSet other)
    cpdef bytearray pack(self)

    @staticmethod
    cdef ScoreSet _create(double glycopeptide_score, double peptide_score, double glycan_score, double glycan_coverage)

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