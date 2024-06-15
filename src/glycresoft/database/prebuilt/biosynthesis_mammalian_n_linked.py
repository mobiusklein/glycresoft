import pkg_resources

from glycresoft.database.builder.glycan import (
    GlycanCompositionEnzymeGraph, ExistingGraphGlycanHypothesisSerializer, GlycanTypes)
from glycresoft.database.prebuilt.utils import hypothesis_register, BuildBase

from io import StringIO


def load_graph():
    resource_bytes = pkg_resources.resource_string(
        __name__,
        "data/mammalian_composition_enzyme_graph.json")
    resource_buffer = StringIO(resource_bytes.decode('utf8'))
    return GlycanCompositionEnzymeGraph.load(resource_buffer)


hypothesis_metadata = {
    "name": 'Biosynthesis Mammalian N-Glycans',
    "hypothesis_type": 'glycan_composition',
    "description": "A collection of biosynthetically feasible *N*-glycans using enzymes commonly found in mammals"
    " and limited to at most 26 monosaccharides. `GnTE` is explicitly omitted to make the space tractable."
}


@hypothesis_register(hypothesis_metadata['name'])
class BiosynthesisMammalianNGlycansBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def build(self, database_connection, **kwargs):
        if kwargs.get('hypothesis_name') is None:
            kwargs['hypothesis_name'] = (self.hypothesis_metadata['name'])
        task = ExistingGraphGlycanHypothesisSerializer(
            load_graph(), database_connection, glycan_classification=GlycanTypes.n_glycan,
            **kwargs)
        task.start()
        return task
