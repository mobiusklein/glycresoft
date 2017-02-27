from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import (
    GlycanComposition as DBGlycanComposition,
    GlycanCombination, GlycanCombinationGlycanComposition)
from glycan_profiling.serialize import (
    DatabaseBoundOperation, Analysis, GlycanCompositionChromatogram,
    IdentifiedGlycopeptide, Glycopeptide)

from .glycan_source import (
    GlycanHypothesisSerializerBase, formula, GlycanCompositionToClass)

from glypy import GlycanComposition


class GlycanAnalysisHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, analysis_id, hypothesis_name, output_connection=None):
        if output_connection is None:
            output_connection = database_connection
        self.output_connection = DatabaseBoundOperation(output_connection)
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.analysis_id = analysis_id
        self.seen_cache = set()

    def extract_composition(self, glycan_chromatogram):
        composition = GlycanComposition.parse(glycan_chromatogram.glycan_composition.serialize())
        if str(composition) in self.seen_cache:
            return
        self.seen_cache.add(str(composition))
        mass = composition.mass()
        composition_string = composition.serialize()
        formula_string = formula(composition.total_composition())
        inst = DBGlycanComposition(
            calculated_mass=mass, formula=formula_string,
            composition=composition_string,
            hypothesis_id=self.hypothesis_id)

        self.output_connection.session.add(inst)
        self.output_connection.session.flush()
        db_obj = self.query(DBGlycanComposition).get(glycan_chromatogram.composition.id)
        for sc in db_obj.structure_classes:
            self.output_connection.session.execute(
                GlycanCompositionToClass.insert(), dict(glycan_id=inst.id, class_id=sc.id))
        self.output_connection.session.flush()

    def run(self):
        q = self.session.query(GlycanCompositionChromatogram).filter(
            GlycanCompositionChromatogram.analysis_id == self.analysis_id)
        for gc in q:
            self.extract_composition(gc)
        self.output_connection.session.commit()


class GlycopeptideAnalysisGlycanCompositionExtractionHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, analysis_id, hypothesis_name):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.analysis_id = analysis_id
        self.seen_cache = set()

    def get_all_compositions(self):
        return self.query(DBGlycanComposition).join(GlycanCombinationGlycanComposition).join(
            Glycopeptide,
            GlycanCombinationGlycanComposition.c.combination_id == Glycopeptide.glycan_combination_id).join(
            IdentifiedGlycopeptide, IdentifiedGlycopeptide.structure_id == Glycopeptide.id).filter(
            IdentifiedGlycopeptide.analysis_id == self.analysis_id)

    def extract_composition(self, db_obj):
        composition = GlycanComposition.parse(db_obj.composition)
        if str(composition) in self.seen_cache:
            return
        self.seen_cache.add(str(composition))
        mass = composition.mass()
        composition_string = composition.serialize()
        formula_string = formula(composition.total_composition())
        inst = DBGlycanComposition(
            calculated_mass=mass, formula=formula_string,
            composition=composition_string,
            hypothesis_id=self.hypothesis_id)
        self.session.add(inst)
        self.session.flush()
        for sc in db_obj.structure_classes:
            self.session.execute(GlycanCompositionToClass.insert(), dict(glycan_id=inst.id, class_id=sc.id))
        self.session.flush()

    def run(self):
        q = self.get_all_compositions()
        for gc in q:
            self.extract_composition(gc)
        self.session.commit()
