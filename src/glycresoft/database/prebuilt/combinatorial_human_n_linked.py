from glycresoft.database.builder.glycan import CombinatorialGlycanHypothesisSerializer
from glycresoft.database.prebuilt.utils import hypothesis_register, BuildBase
from io import StringIO


combinatorial_source = u'''Hex 3 10
HexNAc 2 9
Fuc 0 4
NeuAc 0 5

Fuc < HexNAc
HexNAc > (NeuAc) + 1'''

hypothesis_metadata = {
    "name": 'Combinatorial Human N-Glycans',
    "hypothesis_type": 'glycan_composition'
}


# @hypothesis_register(hypothesis_metadata['name'])
class CombinatorialHumanNGlycansBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def build(self, database_connection, **kwargs):
        kwargs.setdefault('hypothesis_name', self.hypothesis_metadata['name'])
        task = CombinatorialGlycanHypothesisSerializer(
            StringIO(combinatorial_source), database_connection,
            **kwargs)
        task.start()
        return task
