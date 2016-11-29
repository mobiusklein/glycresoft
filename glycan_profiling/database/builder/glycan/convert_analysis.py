from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import GlycanComposition as DBGlycanComposition
from glycan_profiling.serialize import DatabaseBoundOperation, Analysis, GlycanCompositionChromatogram

from glycan_profiling.task import TaskBase
from .glycan_source import (
    GlycanHypothesisSerializerBase, formula, GlycanCompositionToClass)

from glypy import GlycanComposition


class GlycanAnalysisHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, analysis_id, hypothesis_name):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.analysis_id = analysis_id
        self.seen_cache = set()

    def extract_composition(self, glycan_chromatogram):
        composition = GlycanComposition.parse(glycan_chromatogram.glycan_composition.serialize())
        if str(composition) in self.seen_cache:
            continue
        self.seen_cache.add(str(composition))
        mass = composition.mass()
        composition_string = composition.serialize()
        formula_string = formula(composition.total_composition())
        inst = DBGlycanComposition(
            calculated_mass=mass, formula=formula_string,
            composition=composition_string,
            hypothesis_id=self.hypothesis_id)
        self.session.add(inst)
        self.flush()
        db_obj = self.query(DBGlycanComposition).get(glycan_chromatogram.composition.id)
        for sc in db_obj.structure_classes:
            self.session.execute(GlycanCompositionToClass.insert(), dict(glycan_id=inst.id, class_id=sc.id))
        self.flush()

    def run(self):
        q = self.session.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id).yield_per(50)
        for gc in q:
            self.extract_composition(gc)
        self.session.commit()


class GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, analysis_id, hypothesis_name):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.analysis_id = analysis_id
        self.seen_cache = set()
