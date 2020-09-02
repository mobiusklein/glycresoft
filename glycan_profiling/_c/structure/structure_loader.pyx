cimport cython

from numpy cimport npy_uint32 as uint32_t, npy_uint64 as uint64_t


@cython.freelist(1000000)
cdef class GlycopeptideDatabaseRecord(object):
    def __init__(self, id, calculated_mass, glycopeptide_sequence, protein_id,
                 start_position, end_position, peptide_mass, hypothesis_id):
        self.id = id
        self.calculated_mass = calculated_mass
        self.glycopeptide_sequence = glycopeptide_sequence
        self.protein_id = protein_id
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_mass = peptide_mass
        self.hypothesis_id = hypothesis_id

    def __repr__(self):
        template = (
            "{self.__class__.__name__}(id={self.id}, calculated_mass={self.calculated_mass}, "
            "glycopeptide_sequence={self.glycopeptide_sequence}, protein_id={self.protein_id}, "
            "start_position={self.start_position}, end_position={self.end_position}, "
            "peptide_mass={self.peptide_mass}, hypothesis_id={self.hypothesis_id}, ")
        return template.format(self=self)

    def __reduce__(self):
        return self.__class__, (self.id, self.calculated_mass, self.glycopeptide_sequence, self.protein_id,
                                self.start_position, self.end_position, self.peptide_mass, self.hypothesis_id)


@cython.freelist(10000000)
cdef class glycopeptide_key(object):
    def __init__(self, start_position, end_position, peptide_id, protein_id, hypothesis_id,
                 glycan_combination_id, structure_type, site_combination_index):
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_id = peptide_id
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id
        self.glycan_combination_id = glycan_combination_id
        self.structure_type = structure_type
        self.site_combination_index = site_combination_index

    @staticmethod
    cdef glycopeptide_key _create(uint32_t start_position, uint32_t end_position, uint64_t peptide_id,
                                  uint32_t protein_id, uint32_t hypothesis_id, uint64_t glycan_combination_id,
                                  object structure_type, uint64_t site_combination_index):
        cdef glycopeptide_key self = glycopeptide_key.__new__(glycopeptide_key)
        self.start_position = start_position
        self.end_position = end_position
        self.peptide_id = peptide_id
        self.protein_id = protein_id
        self.hypothesis_id = hypothesis_id
        self.glycan_combination_id = glycan_combination_id
        self.structure_type = structure_type
        self.site_combination_index = site_combination_index
        return self

    cpdef glycopeptide_key copy(self):
        return glycopeptide_key._create(
            self.start_position, self.end_position, self.peptide_id, self.protein_id,
            self.hypothesis_id, self.glycan_combination_id, self.structure_type,
            self.site_combination_index)


@cython.binding(True)
cpdef tuple peptide_backbone_fragment_key(self, target, args, dict kwargs):
    key = ("get_fragments", args, frozenset(kwargs.items()))
    return key
