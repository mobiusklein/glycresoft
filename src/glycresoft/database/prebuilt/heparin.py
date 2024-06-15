from glycresoft.database.builder.glycan import CombinatorialGlycanHypothesisSerializer
from glycresoft.database.prebuilt.utils import hypothesis_register, BuildBase
from io import StringIO

combinatorial_source = u'''HexA  1   9
HexN    1   9
enHexA  0   1
@acetyl 0   9
@sulfate    0   48

@acetyl <=  HexN
@sulfate <= ((HexN * 3 - @acetyl) + (HexA * 2) + (enHexA * 2))
HexA <= HexN + 1
HexN <= HexA + 1
'''

hypothesis_metadata = {
    "name": 'Low Molecular Weight Heparins',
    "hypothesis_type": 'glycan_composition',
    "description": 'A database of heparin and heparan sulfate chains covering'
    ' the combinatorial space of possible saccharides of up to 18 monosaccharides.'
}


@hypothesis_register(hypothesis_metadata['name'])
class LowMolecularWeightHeparansBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def build(self, database_connection, **kwargs):
        kwargs.setdefault('hypothesis_name', self.hypothesis_metadata['name'])
        task = CombinatorialGlycanHypothesisSerializer(
            StringIO(combinatorial_source), database_connection,
            **kwargs)
        task.start()
        return task
