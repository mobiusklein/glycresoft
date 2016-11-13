from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import GlycanComposition as DBGlycanComposition
from glycan_profiling.serialize import DatabaseBoundOperation, Analysis, GlycanCompositionChromatogram

from glycan_profiling.task import TaskBase
from .glycan_source import GlycanHypothesisSerializerBase, formula

from glypy import GlycanComposition


class GlycanAnalysisHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, analysis_id, hypothesis_name):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.analysis_id = analysis_id

    def extract_composition(self, glycan_chromatogram):
        composition = GlycanComposition.parse(glycan_chromatogram.glycan_composition.serialize())
        mass = composition.mass()
        composition_string = composition.serialize()
        formula_string = formula(composition.total_composition())
        inst = DBGlycanComposition(
            calculated_mass=mass, formula=formula_string,
            composition=composition_string,
            hypothesis_id=self.hypothesis_id)
        self.session.add(inst)

    def run(self):
        q = self.session.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).yield_per(50)
        for gc in q:
            self.extract_composition(gc)
        self.session.commit()
