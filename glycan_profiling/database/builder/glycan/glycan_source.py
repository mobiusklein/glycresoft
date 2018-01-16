import re

from uuid import uuid4

from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import (
    GlycanComposition as DBGlycanComposition, GlycanClass,
    GlycanCompositionToClass, GlycanTypes)

from glycan_profiling.serialize import DatabaseBoundOperation, func

from glycan_profiling.database.builder.base import HypothesisSerializerBase

from glypy.composition import composition_transform, formula
from glypy.structure import glycan_composition
from glypy import ReducedEnd


class TextFileGlycanCompositionLoader(object):
    def __init__(self, file_object):
        self.file_object = file_object
        self.current_line = 0

    def _partition_line(self, line):
        n = len(line)
        brace_close = line.index("}")
        parts = []
        # line is just glycan composition
        if brace_close == n - 1:
            parts.append(line)
            return parts

        offset = brace_close + 1
        i = 0
        while (offset + i) < n:
            c = line[offset + i]
            if c.isspace():
                break
            i += 1
        parts.append(line[:offset + i])
        parts.extend(t for t in re.split(r"(?:\t|\s{2,})", line[offset + i:]) if t)
        return parts

    def _process_line(self):
        line = next(self.file_object)
        line = line.strip()
        self.current_line += 1
        try:
            while line == '':
                line = next(self.file_object)
                line = line.strip()
            tokens = self._partition_line(line)
            if len(tokens) == 1:
                composition, structure_classes = tokens[0], ()
            elif len(tokens) == 2:
                composition, structure_classes = tokens
                structure_classes = [structure_classes]
            else:
                composition = tokens[0]
                structure_classes = tokens[1:]
            gc = glycan_composition.GlycanComposition.parse(composition)
        except StopIteration:
            raise
        except Exception as e:
            raise Exception("Parsing Error %r occurred at line %d" % (e, self.current_line))
        return gc, structure_classes

    def __next__(self):
        return self._process_line()

    next = __next__

    def __iter__(self):
        return self


_default_label_map = {
    "n glycan": GlycanTypes.n_glycan,
    "n linked": GlycanTypes.n_glycan,
    "o glycan": GlycanTypes.o_glycan,
    "o linked": GlycanTypes.o_glycan,
    "gag linker": GlycanTypes.gag_linker
}


def normalize_lookup(string):
    normalized_string = string.lower().replace("-", " ").replace(
        "\"", '').replace("'", '').strip().rstrip()
    if normalized_string in _default_label_map:
        return _default_label_map[normalized_string]
    else:
        return string


class GlycanClassLoader(object):
    def __init__(self, session):
        self.session = session
        self.store = dict()

    def get(self, name):
        name = normalize_lookup(name)
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


named_reductions = {
    'reduced': 'H2',
    'deuteroreduced': 'HH[2]',
    '2ab': "C7H8N2",
    '2aa': "C7H7NO"
}


named_derivatizations = {
    "permethylated": "methyl",
    "peracetylated": "acetyl"
}


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
    def __init__(self, database_connection, hypothesis_name=None, uuid=None):
        if uuid is None:
            uuid = str(uuid4().hex)
        DatabaseBoundOperation.__init__(self, database_connection)
        self._hypothesis_name = hypothesis_name
        self._hypothesis_id = None
        self._hypothesis = None
        self._structure_class_loader = None
        self.uuid = uuid

    def make_structure_class_loader(self):
        return GlycanClassLoader(self.session)

    @property
    def structure_class_loader(self):
        if self._structure_class_loader is None:
            self._structure_class_loader = self.make_structure_class_loader()
        return self._structure_class_loader

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
        self.loader = TextFileGlycanCompositionLoader(open(self.glycan_file))
        self.transformer = GlycanTransformer(self.loader, self.reduction, self.derivatization)

    def run(self):
        self.make_pipeline()
        structure_class_lookup = self.structure_class_loader
        self.log("Loading Glycan Compositions from Stream for %r" % self.hypothesis)

        acc = []
        counter = 0
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
            counter += 1
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
        self.log("Generated %d glycan compositions" % counter)


class GlycanCompositionHypothesisMerger(GlycanHypothesisSerializerBase):
    def __init__(self, database_connection, source_hypothesis_ids, hypothesis_name, uuid=None):
        GlycanHypothesisSerializerBase.__init__(
            self, database_connection, hypothesis_name, uuid=uuid)
        self.source_hypothesis_ids = source_hypothesis_ids

    def stream_from_hypotheses(self, connection, hypothesis_id):
        self.log("Streaming from %s for hypothesis %d" % (connection, hypothesis_id))
        connection = DatabaseBoundOperation(connection)
        session = connection.session()
        for db_composition in session.query(DBGlycanComposition).filter(
                DBGlycanComposition.hypothesis_id == hypothesis_id):
            structure_classes = list(db_composition.structure_classes)
            if len(structure_classes) > 0:
                yield db_composition, [sc.name for sc in db_composition.structure_classes]
            else:
                yield db_composition, [None]

    def extract_iterative(self):
        for connection, hypothesis_id in self.source_hypothesis_ids:
            for pair in self.stream_from_hypotheses(connection, hypothesis_id):
                yield pair

    def run(self):
        structure_class_lookup = self.structure_class_loader
        acc = []
        seen = set()
        composition_cache = dict()
        counter = 0
        for db_composition, structure_classes in self.extract_iterative():
            counter += 1
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
                if structure_class is None:
                    continue
                structure_class = structure_class_lookup[structure_class]
                acc.append(dict(glycan_id=inst.id, class_id=structure_class.id))
                if (len(acc) % 10000) == 0:
                    self.session.execute(GlycanCompositionToClass.insert(), acc)
                    acc = []
        if acc:
            self.session.execute(GlycanCompositionToClass.insert(), acc)
            acc = []
        self.session.commit()
        assert len(seen) > 0
