from libc.stdint cimport uint32_t, uint64_t

cdef class GlycopeptideDatabaseRecord(object):
    cdef:
        public size_t id
        public double calculated_mass
        public str glycopeptide_sequence
        public int protein_id
        public int start_position
        public int end_position
        public double peptide_mass
        public int hypothesis_id

cdef class glycopeptide_key(object):
    cdef:
        public uint32_t start_position
        public uint32_t end_position
        public uint64_t peptide_id
        public uint32_t protein_id
        public uint32_t hypothesis_id
        public uint64_t glycan_combination_id
        public object structure_type
        public uint64_t site_combination_index

    @staticmethod
    cdef glycopeptide_key _create(uint32_t start_position, uint32_t end_position, uint64_t peptide_id,
                                  uint32_t protein_id, uint32_t hypothesis_id, uint64_t glycan_combination_id,
                                  object structure_type, uint64_t site_combination_index)

    cpdef glycopeptide_key copy(self)