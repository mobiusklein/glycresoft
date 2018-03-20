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