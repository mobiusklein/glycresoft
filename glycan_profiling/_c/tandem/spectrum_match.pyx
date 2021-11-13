cimport cython
# cimport numpy as np
# import numpy as np

# np.import_array()



@cython.freelist(100000)
@cython.final
cdef class ScoreSet(object):
    def __init__(self, glycopeptide_score=0., peptide_score=0., glycan_score=0., glycan_coverage=0., stub_glycopeptide_intensity_utilization=0., n_stub_glycopeptide_matches=0):
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage
        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches

    cpdef bytearray pack(self):
        cdef:
            float[6] data
        data[0] = self.glycopeptide_score
        data[1] = self.peptide_score
        data[2] = self.glycan_score
        data[3] = self.glycan_coverage
        data[4] = self.stub_glycopeptide_intensity_utilization
        data[5] = self.n_stub_glycopeptide_matches
        return ((<char*>data)[:sizeof(float) * 6])

    @staticmethod
    def unpack(bytearray data):
        cdef:
            float* buff
            char* temp
            int n_stub_glycopeptide_matches
        temp = data
        buff = <float*>(temp)
        n_stub_glycopeptide_matches = <int>buff[5]
        return ScoreSet._create(buff[0], buff[1], buff[2], buff[3], buff[4], n_stub_glycopeptide_matches)

    @staticmethod
    cdef ScoreSet _create(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                          float stub_glycopeptide_intensity_utilization, int n_stub_glycopeptide_matches):
        cdef:
            ScoreSet self
        self = ScoreSet.__new__(ScoreSet)
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage
        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches
        return self

    def __eq__(self, other):
        return self._eq(other)

    def __ne__(self, other):
        return not self._eq(other)

    def __lt__(self, other):
        return self._lt(other)

    def __gt__(self, other):
        return self._gt(other)

    def __repr__(self):
        template = (
            "{self.__class__.__name__}({self.glycopeptide_score}, {self.peptide_score},"
            " {self.glycan_score}, {self.glycan_coverage})")
        return template.format(self=self)

    def __len__(self):
        return 4

    def __getitem__(self, int i):
        if i == 0:
            return self.glycopeptide_score
        elif i == 1:
            return self.peptide_score
        elif i == 2:
            return self.glycan_score
        elif i == 3:
            return self.glycan_coverage
        else:
            raise IndexError(i)

    def __reduce__(self):
        return self.__class__, (self.glycopeptide_score, self.peptide_score,
                                self.glycan_score, self.glycan_coverage)
    @classmethod
    def from_spectrum_matcher(cls, match):
        utilization, count = match.count_peptide_Y_ion_utilization()
        return cls(match.score,
            match.peptide_score(),
            match.glycan_score(),
            match.glycan_coverage(),
            utilization, count)

    cpdef bint _eq(self, ScoreSet other):
        if abs(self.glycopeptide_score - other.glycopeptide_score) > 1e-3:
            return False
        if abs(self.peptide_score - other.peptide_score) > 1e-3:
            return False
        if abs(self.glycan_score - other.glycan_score) > 1e-3:
            return False
        if abs(self.glycan_coverage - other.glycan_coverage) > 1e-3:
            return False
        return True

    cpdef bint _lt(self, ScoreSet other):
        if self.glycopeptide_score < other.glycopeptide_score:
            return True
        elif abs(self.glycopeptide_score - other.glycopeptide_score) > 1e-3:
            return False

        if self.peptide_score < other.peptide_score:
            return True
        elif abs(self.peptide_score - other.peptide_score) > 1e-3:
            return False

        if self.glycan_score < other.glycan_score:
            return True
        elif abs(self.glycan_score - other.glycan_score) > 1e-3:
            return False

        if self.glycan_coverage < other.glycan_coverage:
            return True
        return False

    cpdef bint _gt(self, ScoreSet other):
        if self.glycopeptide_score > other.glycopeptide_score:
            return True
        elif abs(self.glycopeptide_score - other.glycopeptide_score) > 1e-3:
            return False

        if self.peptide_score > other.peptide_score:
            return True
        elif abs(self.peptide_score - other.peptide_score) > 1e-3:
            return False

        if self.glycan_score > other.glycan_score:
            return True
        elif abs(self.glycan_score - other.glycan_score) > 1e-3:
            return False

        if self.glycan_coverage > other.glycan_coverage:
            return True
        return False


@cython.freelist(100000)
@cython.final
cdef class FDRSet(object):

    def __init__(self, total_q_value, peptide_q_value, glycan_q_value, glycopeptide_q_value):
        self.total_q_value = total_q_value
        self.peptide_q_value = peptide_q_value
        self.glycan_q_value = glycan_q_value
        self.glycopeptide_q_value = glycopeptide_q_value

    def __eq__(self, other):
        return self._eq(other)

    def __ne__(self, other):
        return not self._eq(other)

    def __lt__(self, other):
        return self._lt(other)

    def __gt__(self, other):
        return self._gt(other)

    def __repr__(self):
        template = (
            "{self.__class__.__name__}({self.total_q_value}, {self.peptide_q_value}, "
            "{self.glycan_q_value}, {self.glycopeptide_q_value})")
        return template.format(self=self)

    cpdef bint _eq(self, FDRSet other):
        if other is None:
            return False
        if self is None:
            return False
        if abs(self.total_q_value - other.total_q_value) > 1e-3:
            return False
        if abs(self.peptide_q_value - other.peptide_q_value) > 1e-3:
            return False
        if abs(self.glycan_q_value - other.glycan_q_value) > 1e-3:
            return False
        if abs(self.glycopeptide_q_value - other.glycopeptide_q_value) > 1e-3:
            return False
        return True

    cpdef bint _lt(self, FDRSet other):
        if self.total_q_value < other.total_q_value:
            return True
        elif abs(self.total_q_value - other.total_q_value) > 1e-3:
            return False

        if self.peptide_q_value < other.peptide_q_value:
            return True
        elif abs(self.peptide_q_value - other.peptide_q_value) > 1e-3:
            return False

        if self.glycan_q_value < other.glycan_q_value:
            return True
        elif abs(self.glycan_q_value - other.glycan_q_value) > 1e-3:
            return False

        if self.glycopeptide_q_value < other.glycopeptide_q_value:
            return True
        return False

    cpdef bint _gt(self, FDRSet other):
        if self.total_q_value > other.total_q_value:
            return True
        elif abs(self.total_q_value - other.total_q_value) > 1e-3:
            return False

        if self.peptide_q_value > other.peptide_q_value:
            return True
        elif abs(self.peptide_q_value - other.peptide_q_value) > 1e-3:
            return False

        if self.glycan_q_value > other.glycan_q_value:
            return True
        elif abs(self.glycan_q_value - other.glycan_q_value) > 1e-3:
            return False

        if self.glycopeptide_q_value > other.glycopeptide_q_value:
            return True
        return False

    @staticmethod
    cdef FDRSet _create(double total_q_value, double peptide_q_value, double glycan_q_value, double glycopeptide_q_value):
        cdef:
            FDRSet self
        self = FDRSet.__new__(FDRSet)
        self.total_q_value = total_q_value
        self.peptide_q_value = peptide_q_value
        self.glycan_q_value = glycan_q_value
        self.glycopeptide_q_value = glycopeptide_q_value
        return self

    cpdef bytearray pack(self):
        cdef:
            double[4] data
        data[0] = self.total_q_value
        data[1] = self.peptide_q_value
        data[2] = self.glycan_q_value
        data[3] = self.glycopeptide_q_value
        return ((<char*>data)[:sizeof(double) * 4])

    @staticmethod
    def unpack(bytearray data):
        cdef:
            double* buff
            char* temp
        temp = data
        buff = <double*>(temp)
        return FDRSet._create(buff[0], buff[1], buff[2], buff[3])

    @classmethod
    def default(cls):
        return FDRSet._create(1.0, 1.0, 1.0, 1.0)

    def __len__(self):
        return 4

    def __getitem__(self, int i):
        if i == 0:
            return self.total_q_value
        elif i == 1:
            return self.peptide_q_value
        elif i == 2:
            return self.glycan_q_value
        elif i == 3:
            return self.glycopeptide_q_value
        else:
            raise IndexError(i)

    def __reduce__(self):
        return self.__class__, (self.total_q_value, self.peptide_q_value,
                                self.glycan_q_value, self.glycopeptide_q_value)