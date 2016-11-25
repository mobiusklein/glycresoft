import re
from collections import Counter
from uuid import uuid4
from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import (
    GlycanComposition as DBGlycanComposition, GlycanClass,
    GlycanCompositionToClass)
from glycan_profiling.serialize import DatabaseBoundOperation

from glycan_profiling.task import TaskBase
from glycan_profiling.database.builder.base import HypothesisSerializerBase

from glypy.composition import glycan_composition, composition_transform, formula
from glypy import ReducedEnd


class GlycanCompositionLoader(object):
    def __init__(self, file_object):
        self.file_object = file_object

    def _process_line(self):
        line = next(self.file_object)
        line = line.strip()
        while line == '':
            line = next(self.file_object)
            line = line.strip()
        tokens = re.split(r"(?:\t|\s{2,})", line)
        if len(tokens) == 1:
            composition, structure_classes = tokens[0], ()
        elif len(tokens) == 2:
            composition, structure_classes = tokens
            structure_classes = [structure_classes]
        else:
            composition = tokens[0]
            structure_classes = tokens[1:]
        gc = glycan_composition.GlycanComposition.parse(composition)
        return gc, structure_classes

    def __next__(self):
        return self._process_line()

    next = __next__

    def __iter__(self):
        return self


class GlycanClassLoader(object):
    def __init__(self, session):
        self.session = session
        self.store = dict()

    def get(self, name):
        try:
            return self.store[name]
        except KeyError:
            glycan_class = self.session.query(GlycanClass).filter(GlycanClass.name == name).first()
            if glycan_class is None:
                glycan_class = GlycanClass(name=name)
                self.session.add(glycan_class)
                self.session.flush()
            self.store[name] = glycan_class
            return glycan_class

    def __getitem__(self, name):
        return self.get(name)


class GlycanTransformer(object):
    def __init__(self, glycan_source, reduction=None, derivatization=None):
        self.glycan_source = glycan_source
        self.reduction = reduction
        self.derivatization = derivatization

        if reduction is not None and isinstance(reduction, str):
            self.reduction = ReducedEnd(reduction)

    def _process_composition(self):
        gc, structure_classes = next(self.glycan_source)
        if self.reduction is not None:
            gc.reducing_end = self.reduction.clone()
        if self.derivatization is not None:
            gc = composition_transform.derivatize(gc, self.derivatization)
        return gc, structure_classes

    def __next__(self):
        return self._process_composition()

    next = __next__

    def __iter__(self):
        return self


class GlycanHypothesisSerializerBase(DatabaseBoundOperation, HypothesisSerializerBase):
    def __init__(self, database_connection, hypothesis_name=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self._hypothesis_name = hypothesis_name
        self._hypothesis_id = None
        self._hypothesis = None
        self.uuid = str(uuid4().hex)

    def structure_class_loader(self):
        return GlycanClassLoader(self.session)

    def _construct_hypothesis(self):
        if self._hypothesis_name is None:
            self._hypothesis_name = self._make_name()
        self._hypothesis = GlycanHypothesis(name=self._hypothesis_name, uuid=self.uuid)

        self.session.add(self._hypothesis)
        self.session.commit()

        self._hypothesis_id = self._hypothesis.id
        self._hypothesis_name = self._hypothesis.name

    def _make_name(self):
        return "GlycanHypothesis-" + self.uuid


class TextFileGlycanHypothesisSerializer(GlycanHypothesisSerializerBase):
    def __init__(self, glycan_text_file, database_connection, reduction=None, derivatization=None,
                 hypothesis_name=None):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)

        self.glycan_file = glycan_text_file
        self.reduction = reduction
        self.derivatization = derivatization

        self.loader = None
        self.transformer = None

    def make_pipeline(self):
        self.loader = GlycanCompositionLoader(open(self.glycan_file))
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader()
        self.log("Loading Glycan Compositions from File for %r" % self.hypothesis)

        acc = []
        for composition, structure_classes in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            self.session.add(inst)
            self.session.flush()
            for structure_class in structure_classes:
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()


class GlycanCompositionHypothesisMerger(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, source_hypothesis_ids, hypothesis_name):
        GlycanHypothesisSerializerBase.__init__(self, database_connection, hypothesis_name)
        self.source_hypothesis_ids = source_hypothesis_ids

    def extract_iterative(self):
        for hypothesis_id in self.source_hypothesis_ids:
            q = self.session.query(DBGlycanComposition).filter(
                DBGlycanComposition.hypothesis_id == hypothesis_id)
            for db_composition in q:
                yield db_composition, [sc.name for sc in db_composition.structure_classes]

    def run(self):
        structure_class_lookup = self.structure_class_loader()
        self.log("Merging Glycan Composition Lists for %r" % self.hypothesis)

        acc = []
        seen = set()
        composition_cache = dict()
        for db_composition, structure_classes in self.extract_iterative():
            composition_string = db_composition.composition
            novel_combinations = set()
            for structure_class in structure_classes:
                if (composition_string, structure_class) not in seen:
                    novel_combinations.add(structure_class)
            if len(novel_combinations) == 0:
                continue
            try:
                inst = composition_cache[composition_string]
            except KeyError:
                mass = db_composition.calculated_mass
                formula_string = db_composition.formula
                inst = DBGlycanComposition(
                    calculated_mass=mass, formula=formula_string,
                    composition=composition_string,
                    hypothesis_id=self.hypothesis_id)
                self.session.add(inst)
                self.session.flush()
                composition_cache[composition_string] = inst
            for structure_class in novel_combinations:
                seen.add((composition_string, structure_class))
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if len(acc) % 100 == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
            if acc:
                self.session.execute(GlycanCompositionToClass.insert(), acc)
                acc = []
            self.session.commit()