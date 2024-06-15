import itertools

from glypy.composition import Composition
from glypy.composition.composition_transform import strip_derivatization
from glypy.structure.glycan_composition import FrozenMonosaccharideResidue, MonosaccharideResidue
from glypy.io.nomenclature.identity import is_a

from glycresoft.structure.fragment_match_map import SpectrumGraph
from glycresoft.structure.denovo import MassWrapper, PathFinder


class DeNovoSolution(object):
    def __init__(self, scan, supporting_paths=None, completion_gap=None):
        self.scan = scan
        self.supporting_paths = supporting_paths
        self.completion_gap = completion_gap

    @property
    def precursor_mass(self):
        return self.scan.precursor_information.neutral_mass


class DeNovoGlycanSequencer(PathFinder):
    def __init__(self, spectrum, components, precursor_error_tolerance=1e-5, product_error_tolerance=2e-5):
        super(DeNovoGlycanSequencer, self).__init__(spectrum, components, product_error_tolerance)
        self.precursor_error_tolerance = precursor_error_tolerance
