import os
from glypy.io import glycoct, glyspace
from glypy.structure.glycan import NamedGlycan
from glypy.composition.glycan_composition import GlycanComposition

from glycan_profiling.serialize import (
    GlycanComposition as DBGlycanComposition,
    GlycanStructure,
    GlycanHypothesis)

from .glycan_source import (
    GlycanTransformer, GlycanHypothesisSerializerBase,
    formula)


config = {
    "transient_store_directory": None
}


def set_transient_store(path):
    config["transient_store_directory"] = path

def get_transient_store():
    return config["transient_store_directory"]    


class GlycanCompositionSerializationCache(object):
    def __init__(self, session, hypothesis_id):
        self.session = session
        self.store = dict()
        self.hypothesis_id = hypothesis_id

    def get(self, composition_string):
        try:
            return self.store[composition_string]
        except KeyError:
            composition = GlycanCompostion.parse(composition_string)
            mass = composition.mass()
            formula_string = formula(composition.total_composition())
            db_obj = DBGlycanComposition(
                calculated_mass=mass,
                composition=composition_string,
                formula=formula_string,
                hypothesis_id=self.hypothesis_id)
            self.store[composition_string] = db_obj
            self.session.add(db_obj)
            self.session.flush()
            return db_obj

    def __getitem__(self, k):
        return self.get(k)


class GlyspaceGlycanStructureHypothesisSerializerBase(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, hypothesis_name=None, reduction=None, derivatization=None):
        super(GlyspaceGlycanStructureHypothesisSerializer, self).__init__(database_connection, hypothesis_name)
        self.reduction = reduction
        self.derivatization = derivatization
        self.loader = None
        self.transformer = None

    def _get_sparql(self):
        raise NotImplementedError()

    def _get_store_name(self):
        raise NotImplementedError()

    def _store_result(self, response):
        dir_path = get_transient_store()
        if dir_path is not None:
            file_path = os.path.join(dir_path, self._get_store_name())
            response.serialize(file_path)

    def execute_query(self):
        sparql = self._get_sparql()
        response = glyspace.query(sparql)
        self._store_result(response)
        return response

    def translate_response(self, response):
        for name, glycosequence in response:
            try:
                structure = glycoct.loads(glycosequence, structure_class=NamedGlycan)
                structure.name = name
                yield structure
            except Exception as e:
                self.error("Error in translate_response", e)
                continue

    def load_structures(self):
        self.log("Querying GlySpace")
        sparql_response = self.execute_query()
        self.log("Translating Response")
        return self.translate_response(sparql_response)

    def make_pipeline(self):
        self.loader = self.load_structures()
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        composition_cache = GlycanCompositionSerializationCache(self.session, self.hypothesis_id)
        i = 0
        last = 0
        interval = 100
        for structure in self.transformer:
            mass = structure.mass()
            structure_string = structure.serialize()
            formula_string = formula(structure.total_composition())
            composition_string = glypy.GlycanCompostion.from_glycan(structure).serialize()
            structure = GlycanStructure(
                glycan_sequence=structure_string,
                formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id,
                calculated_mass=mass)
            composition_obj = composition_cache[composition_string]
            structure.composition_id = composition_obj.id
            self.session.add(structure)
            i += 1
            if i - last > interval:
                self.session.commit()
                last = i
        self.session.commit()

