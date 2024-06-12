from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

cdef class SignatureSpecification(object):
    cdef:
        public tuple components
        public tuple masses
        public bint _is_compound
        public Py_hash_t _hash
        double* _masses
        size_t n_masses

    cdef void _init_mass_array(self)
    cpdef bint is_expected(self, glycan_composition)
    cpdef int count_of(self, glycan_composition)
    cpdef DeconvolutedPeak peak_of(self, DeconvolutedPeakSet spectrum, double error_tolerance)


cdef class SignatureIonMatchRecord:
    cdef:
        public dict expected_matches
        public dict unexpected_matches
        public object glycan_composition
        public double score

    @staticmethod
    cdef SignatureIonMatchRecord _create(dict expected_matches, dict unexpected_matches, object glycan_composition)
    cpdef match(self, list signatures, DeconvolutedPeakSet spectrum, double error_tolerance=*)


cdef class SignatureIonIndexMatch:
    cdef:
        public dict glycan_to_key
        public dict key_to_record
        public double base_peak_intensity

    @staticmethod
    cdef SignatureIonIndexMatch _create(dict glycan_to_key, dict key_to_record, double base_peak_intensity)

    cpdef SignatureIonMatchRecord record_for(self, object glycan_composition)


cdef class SignatureIonIndex:
    cdef:
        public list signatures
        public dict glycan_to_key
        public dict key_to_representative

    cpdef SignatureIonIndexMatch match(self, DeconvolutedPeakSet spectrum, double error_tolerance=*)



cdef class OxoniumIndexMatch(object):
    cdef:
        public dict index_matches
        public dict glycan_to_index
        public dict id_to_index

    @staticmethod
    cdef OxoniumIndexMatch _create(dict index_matches, dict glycan_to_index, dict id_to_index)

    cpdef list by_glycan(self, glycan)
    cpdef list by_id(self, glycan_id)


cdef class OxoniumIndex(object):
    '''An index for quickly matching all oxonium ions against a spectrum and efficiently mapping them
    back to individual glycan compositions.
    '''
    cdef:
        public list fragments
        public dict fragment_index
        public dict glycan_to_index
        public dict index_to_glycan
        public dict index_to_simplified_index

    cpdef object simplify(self)
    cpdef OxoniumIndexMatch match(self, DeconvolutedPeakSet spectrum, double error_tolerance)