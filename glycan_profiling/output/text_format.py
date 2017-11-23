from glypy.io import iupac, glycoct
from glypy import Glycan


from glycan_profiling.task import TaskBase


def build_partial_subgraph_n_glycan(composition):
    composition = composition.clone()
    if composition["HexNAc"] < 2:
        raise ValueError("Not enough HexNAc present to extract N-glycan core")
    root = iupac.loads("?-D-Glcp2NAc").clone(prop_id=False)
    composition['HexNAc'] -= 1
    b2 = iupac.loads("b-D-Glcp2NAc").clone(prop_id=False)
    root.add_monosaccharide(b2, position=4, child_position=1)
    composition['HexNAc'] -= 1
    if composition['Hex'] < 3:
        raise ValueError("Not enough Hex present to extract N-glycan core")
    composition['Hex'] -= 3
    b3 = iupac.loads("b-D-Manp").clone()
    b2.add_monosaccharide(b3, position=4, child_position=1)
    b4 = iupac.loads("a-D-Manp").clone()
    b3.add_monosaccharide(b4, position=3, child_position=1)
    b5 = iupac.loads("a-D-Manp").clone()
    b3.add_monosaccharide(b5, position=6, child_position=1)
    subgraph = Glycan(root, index_method=None)
    return composition, subgraph


def to_glycoct_partial_n_glycan(composition):
    a, b = build_partial_subgraph_n_glycan(composition)
    writer = glycoct.OrderRespectingGlycoCTWriter(b)
    writer.handle_glycan(writer.structure)
    writer.add_glycan_composition(a)
    return writer.buffer.getvalue()


class GlycoCTCompositionListExporter(TaskBase):
    def __init__(self, outstream, glycan_composition_iterable):
        self.outstream = outstream
        self.glycan_composition_iterable = glycan_composition_iterable

    def start(self):
        for glycan_composition in self.glycan_composition_iterable:
            text = to_glycoct_partial_n_glycan(glycan_composition)
            self.outstream.write(text)
            self.outstream.write("\n")
