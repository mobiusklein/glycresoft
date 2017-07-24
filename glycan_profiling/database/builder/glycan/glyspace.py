import logging
try:
    logger_to_silence = logging.getLogger("rdflib")
    logger_to_silence.propagate = False
    logger_to_silence.setLevel("CRITICAL")
except Exception:
    pass

import os
from glypy.io import glycoct, glyspace
from glypy.structure.glycan import NamedGlycan
from glypy.composition.glycan_composition import GlycanComposition

from rdflib.plugins.sparql.results import xmlresults

from glycan_profiling.serialize import (
    GlycanComposition as DBGlycanComposition,
    GlycanClass,
    GlycanStructure,
    GlycanHypothesis,
    GlycanStructureToClass, GlycanCompositionToClass)

from .glycan_source import (
    GlycanTransformer, GlycanHypothesisSerializerBase,
    formula)

from glycan_profiling.task import TaskBase


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
            match = self.store[composition_string]
            return match
        except KeyError:
            composition = GlycanComposition.parse(composition_string)
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


o_glycan_query = """
SELECT DISTINCT ?saccharide ?glycoct ?taxon ?motif WHERE {
    ?saccharide a glycan:saccharide .
    ?saccharide glycan:has_glycosequence ?sequence .
    ?saccharide skos:exactMatch ?gdb .
    ?gdb glycan:has_reference ?ref .
    ?ref glycan:is_from_source ?source .
    ?source glycan:has_taxon ?taxon
    FILTER CONTAINS(str(?sequence), "glycoct") .
    ?sequence glycan:has_sequence ?glycoct .
    ?saccharide glycan:has_motif ?motif .
    FILTER(?motif in (glycoinfo:G00032MO, glycoinfo:G00042MO, glycoinfo:G00034MO,
                      glycoinfo:G00036MO, glycoinfo:G00033MO, glycoinfo:G00038MO,
                      glycoinfo:G00040MO, glycoinfo:G00037MO, glycoinfo:G00044MO))
}
"""

restricted_o_glycan_query = """
SELECT DISTINCT ?saccharide ?glycoct ?taxon ?motif WHERE {
    ?saccharide a glycan:saccharide .
    ?saccharide glycan:has_glycosequence ?sequence .
    ?saccharide skos:exactMatch ?gdb .
    ?gdb glycan:has_reference ?ref .
    ?ref glycan:is_from_source ?source .
    ?source glycan:has_taxon ?taxon
    FILTER CONTAINS(str(?sequence), "glycoct") .
    ?sequence glycan:has_sequence ?glycoct .
    ?saccharide glycan:has_motif ?motif .
    FILTER(?motif in (glycoinfo:G00032MO, glycoinfo:G00034MO))
}
"""


n_glycan_query = """
SELECT DISTINCT ?saccharide ?glycoct ?taxon ?motif WHERE {
    ?saccharide a glycan:saccharide .
    ?saccharide glycan:has_glycosequence ?sequence .
    ?saccharide skos:exactMatch ?gdb .
    ?gdb glycan:has_reference ?ref .
    ?ref glycan:is_from_source ?source .
    ?source glycan:has_taxon ?taxon
    FILTER CONTAINS(str(?sequence), "glycoct") .
    ?sequence glycan:has_sequence ?glycoct .
    ?saccharide glycan:has_motif ?motif .
    FILTER(?motif in (glycoinfo:G00026MO))
}
"""


taxonomy_query_template = """
PREFIX up:<http://purl.uniprot.org/core/>
PREFIX taxondb:<http://purl.uniprot.org/taxonomy/>
SELECT ?taxon
FROM <http://sparql.uniprot.org/taxonomy>
WHERE
{
        ?taxon a up:Taxon .
        {
            ?taxon rdfs:subClassOf taxondb:%(taxon)s .
        }
}
"""


motif_to_class_map = {
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00032MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00042MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00034MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00036MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00033MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00038MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00040MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00037MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00044MO"): "O-Linked",
    glyspace.URIRef("http://rdf.glycoinfo.org/glycan/G00026MO"): "N-Linked"
}


def parse_taxon(taxon_uri):
    uri = str(taxon_uri)
    taxon = uri.split("/")[-1].replace(".rdf", "")
    return taxon


class TaxonomyFilter(object):
    def __init__(self, target_taxonomy, include_children=True):
        if isinstance(target_taxonomy, (tuple, list, set)):
            self.target_taxonomy = str(target_taxonomy)
            self.include_children = include_children
            self.filter_set = set(target_taxonomy)
        else:
            self.target_taxonomy = str(target_taxonomy)
            self.include_children = include_children
            self.filter_set = None
            self.prepare_filter()

    def query_all_taxa(self):
        query = taxonomy_query_template % {"taxon": self.target_taxonomy}
        up = glyspace.UniprotRDFClient()
        result = up.query(query)
        return {parse_taxon(o['taxon']) for o in result} | {str(self.target_taxonomy)}

    def prepare_filter(self):
        if self.include_children:
            self.filter_set = self.query_all_taxa()
        else:
            self.filter_set = set([self.target_taxonomy])

    def __call__(self, structure, name, taxon, motif):
        return taxon not in self.filter_set

    def __repr__(self):
        if len(self.filter_set) > 10:
            return "TaxonomyFilter({%s, ...})" % (", ".join(map(repr, list(self.filter_set)[:10])),)
        else:
            return "TaxonomyFilter({%s})" % (", ".join(map(repr, self.filter_set)),)


def is_not_human(structure, name, taxon, motif):
    return taxon != "9606"


# TODO: Refactor database builders to include an
# instance of this object to delegate responsibility
# to for actually communicating with GlySpace
class GlyspaceQueryHandler(TaskBase):
    def __init__(self, query, filter_functions=None):
        if filter_functions is None:
            filter_functions = []
        self.sparql = query
        self.filter_functions = filter_functions
        self.response = None

    def query(self):
        return glyspace.query(self.sparql)

    def translate_response(self, response):
        for name, glycosequence, taxon, motif in response:
            taxon = parse_taxon(taxon)
            try:
                structure = glycoct.loads(glycosequence, structure_class=NamedGlycan)
                structure.name = name

                passed = True
                for func in self.filter_functions:
                    if func(structure, name=name, taxon=taxon, motif=motif):
                        passed = False
                        break
                if not passed:
                    continue

                yield structure, motif_to_class_map[motif]
            except glycoct.GlycoCTError as e:
                continue
            except Exception as e:
                self.error("Error in translate_response of %s" % name, e)
                continue

    def execute(self):
        self.response = self.query()
        return self.translate_response(self.response)


class GlyspaceGlycanStructureHypothesisSerializerBase(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, hypothesis_name=None, reduction=None, derivatization=None,
                 filter_functions=None, simplify=False):
        super(GlyspaceGlycanStructureHypothesisSerializerBase, self).__init__(database_connection, hypothesis_name)
        if filter_functions is None:
            filter_functions = []
        self.reduction = reduction
        self.derivatization = derivatization
        self.loader = None
        self.transformer = None
        self.composition_cache = None
        self.seen = dict()
        self.filter_functions = filter_functions
        self.simplify = simplify

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

    def load_query_results(self):
        dir_path = get_transient_store()
        if dir_path is not None:
            file_path = os.path.join(dir_path, self._get_store_name())
            if os.path.exists(file_path):
                p = xmlresults.XMLResultParser()
                results = p.parse(open(file_path))
                return results
            else:
                return None
        else:
            return None

    def translate_response(self, response):
        for name, glycosequence, taxon, motif in response:
            taxon = parse_taxon(taxon)
            try:
                structure = glycoct.loads(glycosequence, structure_class=NamedGlycan)
                structure.name = name

                passed = True
                for func in self.filter_functions:
                    if func(structure, name=name, taxon=taxon, motif=motif):
                        passed = False
                        break
                if not passed:
                    continue

                yield structure, motif_to_class_map[motif]
            except glycoct.GlycoCTError as e:
                continue
            except Exception as e:
                self.error("Error in translate_response of %s" % name, e)
                continue

    def load_structures(self):
        self.log("Querying GlySpace")
        sparql_response = self.load_query_results()
        if sparql_response is None:
            sparql_response = self.execute_query()
        self.log("Translating Response")
        return self.translate_response(sparql_response)

    def make_pipeline(self):
        self.loader = self.load_structures()
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def handle_new_structure(self, structure, motif):
        mass = structure.mass()
        try:
            structure_string = structure.serialize()
        except Exception as e:
            print(structure.name)
            raise e
        formula_string = formula(structure.total_composition())

        glycan_comp = GlycanComposition.from_glycan(structure)
        if self.simplify:
            glycan_comp.drop_configurations()
            glycan_comp.drop_stems()
            glycan_comp.drop_positions()
        composition_string = glycan_comp.serialize()

        structure = GlycanStructure(
            glycan_sequence=structure_string,
            formula=formula_string,
            composition=composition_string,
            hypothesis_id=self.hypothesis_id,
            calculated_mass=mass)
        composition_obj = self.composition_cache[composition_string]
        structure.glycan_composition_id = composition_obj.id
        self.session.add(structure)
        self.session.flush()
        structure_class = self.structure_class_loader[motif]
        rel = dict(glycan_id=structure.id, class_id=structure_class.id)
        self.session.execute(GlycanStructureToClass.insert(), rel)
        self.seen[structure_string] = structure

    def handle_duplicate_structure(self, structure, motif):
        db_structure = self.seen[structure.serialize()]
        structure_class = self.structure_class_loader[motif]
        if structure_class not in db_structure.structure_classes:
            rel = dict(glycan_id=db_structure.id, class_id=structure_class.id)
            self.session.execute(GlycanStructureToClass.insert(), rel)

    def propagate_composition_motifs(self):
        pairs = self.query(
            DBGlycanComposition.id, GlycanClass.id).join(GlycanStructure).join(
            GlycanStructure.structure_classes).filter(
            DBGlycanComposition.hypothesis_id == self.hypothesis_id).group_by(
            DBGlycanComposition.id, GlycanClass.id).all()
        for glycan_id, structure_class_id in pairs:
            self.session.execute(
                GlycanCompositionToClass.insert(),
                dict(glycan_id=glycan_id, class_id=structure_class_id))
        self.session.commit()

    def run(self):
        self.make_pipeline()
        self.composition_cache = GlycanCompositionSerializationCache(self.session, self.hypothesis_id)
        i = 0
        last = 0
        interval = 100
        for structure, motif in self.transformer:
            if structure.serialize() in self.seen:
                self.handle_duplicate_structure(structure, motif)
            else:
                self.handle_new_structure(structure, motif)
            i += 1
            if i - last > interval:
                self.session.commit()
                last = i
        self.session.commit()
        self.propagate_composition_motifs()
        self.log("Stored %d glycan structures and %d glycan compositions" % (
            self.session.query(GlycanStructure).filter(
                GlycanStructure.hypothesis_id == self.hypothesis_id).count(),
            self.session.query(DBGlycanComposition).filter(
                DBGlycanComposition.hypothesis_id == self.hypothesis_id).count()))


class NGlycanGlyspaceHypothesisSerializer(GlyspaceGlycanStructureHypothesisSerializerBase):
    def _get_sparql(self):
        return n_glycan_query

    def _get_store_name(self):
        return "n-glycans-query.sparql.xml"


class OGlycanGlyspaceHypothesisSerializer(GlyspaceGlycanStructureHypothesisSerializerBase):
    def _get_sparql(self):
        return o_glycan_query

    def _get_store_name(self):
        return "o-glycans-query.sparql.xml"


class RestrictedOGlycanGlyspaceHypothesisSerializer(GlyspaceGlycanStructureHypothesisSerializerBase):
    def _get_sparql(self):
        return restricted_o_glycan_query

    def _get_store_name(self):
        return "restricted-o-glycans-query.sparql.xml"
