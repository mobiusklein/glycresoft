cimport cython

from cpython.object cimport PyObject
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem


@cython.freelist(100000)
@cython.final
cdef class ScoreSet(object):
    def __init__(self, glycopeptide_score=0., peptide_score=0., glycan_score=0., glycan_coverage=0.,
                 stub_glycopeptide_intensity_utilization=0., oxonium_ion_intensity_utilization=0.,
                 n_stub_glycopeptide_matches=0, peptide_coverage=0.0):
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage
        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.oxonium_ion_intensity_utilization = oxonium_ion_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches
        self.peptide_coverage = peptide_coverage

    cpdef bytearray pack(self):
        cdef:
            float[7] data
        data[0] = self.glycopeptide_score
        data[1] = self.peptide_score
        data[2] = self.glycan_score
        data[3] = self.glycan_coverage
        data[4] = self.stub_glycopeptide_intensity_utilization
        data[5] = self.oxonium_ion_intensity_utilization
        data[6] = self.n_stub_glycopeptide_matches
        data[7] = self.peptide_coverage
        return ((<char*>data)[:sizeof(float) * 7])

    @staticmethod
    def unpack(bytearray data):
        cdef:
            float* buff
            char* temp
            int n_stub_glycopeptide_matches
            float stub_utilization
            float oxonium_utilization
            float peptide_coverage
        temp = data
        buff = <float*>(temp)
        stub_utilization = buff[4]
        oxonium_utilization = buff[5]
        n_stub_glycopeptide_matches = <int>buff[6]
        peptide_coverage = buff[7]
        return ScoreSet._create(buff[0], buff[1], buff[2], buff[3], stub_utilization,
                                oxonium_utilization, n_stub_glycopeptide_matches, peptide_coverage)

    @staticmethod
    cdef ScoreSet _create(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                          float stub_glycopeptide_intensity_utilization, float oxonium_ion_intensity_utilization,
                          int n_stub_glycopeptide_matches, float peptide_coverage):
        cdef:
            ScoreSet self
        self = ScoreSet.__new__(ScoreSet)
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage
        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.oxonium_ion_intensity_utilization = oxonium_ion_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches
        self.peptide_coverage = peptide_coverage
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
            " {self.glycan_score}, {self.glycan_coverage}, {self.stub_glycopeptide_intensity_utilization},"
            " {self.oxonium_ion_intensity_utilization}, {self.n_stub_glycopeptide_matches}, {self.peptide_coverage})")
        return template.format(self=self)

    def __len__(self):
        return 7

    def __getitem__(self, int i):
        if i == 0:
            return self.glycopeptide_score
        elif i == 1:
            return self.peptide_score
        elif i == 2:
            return self.glycan_score
        elif i == 3:
            return self.glycan_coverage
        elif i == 4:
            return self.stub_glycopeptide_intensity_utilization
        elif i == 5:
            return self.oxonium_ion_intensity_utilization
        elif i == 6:
            return self.n_stub_glycopeptide_matches
        elif i == 7:
            return self.peptide_coverage
        else:
            raise IndexError(i)

    def __iter__(self):
        yield self.glycopeptide_score
        yield self.peptide_score
        yield self.glycan_score
        yield self.glycan_coverage
        yield self.stub_glycopeptide_intensity_utilization
        yield self.oxonium_ion_intensity_utilization
        yield self.n_stub_glycopeptide_matches
        yield self.peptide_coverage

    def __reduce__(self):
        return self.__class__, (self.glycopeptide_score, self.peptide_score,
                                self.glycan_score, self.glycan_coverage, self.stub_glycopeptide_intensity_utilization,
                                self.oxonium_ion_intensity_utilization, self.n_stub_glycopeptide_matches)

    @classmethod
    def from_spectrum_matcher(cls, match):
        stub_utilization, count = match.count_peptide_Y_ion_utilization()
        oxonium_utilization = match.oxonium_ion_utilization()
        return cls(match.score,
            match.peptide_score(),
            match.glycan_score(),
            match.glycan_coverage(),
            stub_utilization,
            oxonium_utilization,
            count)

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


@cython.final
cdef class PeakLabelMap(object):

    def __init__(self, name_to_peaks=None):
        if name_to_peaks is None:
            name_to_peaks = dict()
        self.name_to_peaks = name_to_peaks

    @staticmethod
    cdef PeakLabelMap _create():
        cdef PeakLabelMap self = PeakLabelMap.__new__(PeakLabelMap)
        self.name_to_peaks = {}
        return self

    cdef inline void add(self, str name, DeconvolutedPeak peak):
        cdef:
            list acc
        PyDict_SetItem(self.name_to_peaks, name, peak)

    cdef inline DeconvolutedPeak get(self, str name, bint* found):
        cdef:
            PyObject* temp
            DeconvolutedPeak result

        temp = PyDict_GetItem(self.name_to_peaks, name)
        if temp == NULL:
            found[0] = False
            return None
        else:
            result = <DeconvolutedPeak>temp
            found[0] = True
            return result