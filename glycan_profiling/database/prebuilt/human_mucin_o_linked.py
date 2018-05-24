import pkg_resources

from glycan_profiling.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer)
from glycan_profiling.database.prebuilt.utils import hypothesis_register, BuildBase

from io import StringIO


def load_text():
    resource_bytes = pkg_resources.resource_string(
        __name__,
        "data/mucin_compositions.txt")
    resource_buffer = StringIO(resource_bytes.decode('utf8'))
    return resource_buffer


hypothesis_metadata = {
    "name": 'Biosynthesis Human Mucin O-Glycans',
    "hypothesis_type": 'glycan_composition'
}


@hypothesis_register(hypothesis_metadata['name'])
class BiosynthesisHumanMucinOGlycansBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def build(self, database_connection, **kwargs):
        if kwargs.get('hypothesis_name') is None:
            kwargs['hypothesis_name'] = (self.hypothesis_metadata['name'])
        print(kwargs)
        task = TextFileGlycanHypothesisSerializer(
            load_text(), database_connection,
            **kwargs)
        task.start()
