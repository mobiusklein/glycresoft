from glycan_profiling.database.builder.glycan import CombinatorialGlycanHypothesisSerializer
from glycan_profiling.database.prebuilt.utils import hypothesis_register, BuildBase
from io import StringIO


combinatorial_source = u'''Hex 3 10
HexNAc 2 9
Fuc 0 5
NeuAc 0 5
NeuGc 0 5

Fuc < HexNAc
HexNAc > (NeuAc + NeuGc) + 1'''

hypothesis_metadata = {
    "name": 'Combinatorial Mammalian N-Glycans',
}


@hypothesis_register(hypothesis_metadata['name'])
class CombinatorialMammalianNGlycansBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def build(self, database_connection, **kwargs):
        task = CombinatorialGlycanHypothesisSerializer(
            StringIO(combinatorial_source), database_connection,
            hypothesis_name=hypothesis_metadata['name'])
        task.start()
