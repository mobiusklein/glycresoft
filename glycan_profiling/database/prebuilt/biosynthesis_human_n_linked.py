import pkg_resources

from glycan_profiling.database.builder.glycan import (
    GlycanCompositionEnzymeGraph, ExistingGraphGlycanHypothesisSerializer, GlycanTypes)
from glycan_profiling.database.prebuilt.utils import hypothesis_register, BuildBase

from io import StringIO


def load_graph():
    resource_bytes = pkg_resources.resource_string(
        __name__,
        "data/human_composition_enzyme_graph.json")
    resource_buffer = StringIO(resource_bytes.decode('utf8'))
    return GlycanCompositionEnzymeGraph.load(resource_buffer)


hypothesis_metadata = {
    "name": 'Biosynthesis Human N-Glycans',
    "hypothesis_type": 'glycan_composition'
}


@hypothesis_register(hypothesis_metadata['name'])
class BiosynthesisHumanNGlycansBuilder(BuildBase):

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
