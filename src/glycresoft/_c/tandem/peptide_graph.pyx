cimport cython

from libc.math cimport fabs

from glycresoft._c.chromatogram_tree.mass_shift cimport MassShiftBase
from glycresoft._c.database.mass_collection cimport NeutralMassDatabaseImpl


cdef bint isclose(double x, double y):
    return abs(x - y) < 1e-3


@cython.freelist(10000)
cdef class PeptideMassNode:
    cdef:
        public double peptide_mass
        public MassShiftBase mass_shift
        public double delta_mass
        public str scan_id
        public float glycan_score

    def __init__(self, peptide_mass, mass_shift, delta_mass, scan_id, glycan_score):
        self.peptide_mass = peptide_mass
        self.mass_shift = mass_shift
        self.delta_mass = delta_mass
        self.scan_id = scan_id
        self.glycan_score = glycan_score

    def __repr__(self):
        return f"{self.__class__.__name__}({self.peptide_mass}, {self.mass_shift}, {self.scan_id}, {self.glycan_score})"

    def __eq__(self, other):
        if other is None:
            return False
        if not isclose(self.peptide_mass, other.peptide_mass):
            return False
        if not isclose(self.delta_mass, other.delta_mass):
            return False
        if self.scan_id != other.scan_id:
            return False
        if self.mass_shift != other.mass_shift:
            return False

        return True

    def __ne__(self, other):
        return not self == other


@cython.cdivision(True)
cpdef list intersecting_peptide_masses(NeutralMassDatabaseImpl query_nodes,
                                       NeutralMassDatabaseImpl reference_nodes,
                                       float error_tolerance=5e-6):
    cdef:
        list accumulator
        size_t query_i, reference_i, query_n, reference_n
        size_t checkpoint
        PeptideMassNode query_node, reference_node

    accumulator = []

    query_i = 0
    reference_i = 0
    query_n = len(query_nodes)
    reference_n = len(reference_nodes)

    checkpoint = -1

    while query_i < query_n and reference_i < reference_n:
        query_node = <PeptideMassNode>query_nodes.get_item(query_i)
        reference_node = <PeptideMassNode>reference_nodes.get_item(reference_i)
        if fabs(query_node.peptide_mass - reference_node.peptide_mass) / reference_node.peptide_mass <= error_tolerance:
            accumulator.append((query_node, reference_node))
            if checkpoint == -1:
                checkpoint = reference_i
            query_i += 1
            reference_i += 1
        elif query_node.peptide_mass < reference_node.peptide_mass:
            if checkpoint != -1 and query_i < (query_n - 1) and fabs(query_node.peptide_mass - (
                    <PeptideMassNode>query_nodes.get_item(query_i + 1)).peptide_mass) / query_node.peptide_mass < error_tolerance:
                reference_i = checkpoint
                checkpoint = -1
            query_i += 1
        else:
            reference_i += 1
    return accumulator

