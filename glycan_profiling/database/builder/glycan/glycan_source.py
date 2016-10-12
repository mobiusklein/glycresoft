from uuid import uuid4
from glycan_profiling.serialize.hypothesis import GlycanHypothesis
from glycan_profiling.serialize.hypothesis.glycan import GlycanComposition as DBGlycanComposition
from glycan_profiling.serialize import DatabaseBoundOperation

from glycan_profiling.task import TaskBase

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
        gc = glycan_composition.GlycanComposition.parse(line)
        return gc

    def __next__(self):
        return self._process_line()

    next = __next__

    def __iter__(self):
        return self


class GlycanTransformer(object):
    def __init__(self, glycan_source, reduction=None, derivatization=None):
        self.glycan_source = glycan_source
        self.reduction = reduction
        self.derivatization = derivatization

        if reduction is not None and isinstance(reduction, str):
            self.reduction = ReducedEnd(reduction)

    def _process_composition(self):
        gc = next(self.glycan_source)
        if self.reduction is not None:
            gc.reducing_end = self.reduction.clone()
        if self.derivatization is not None:
            gc = composition_transform.derivatize(gc, self.derivatization)
        return gc

    def __next__(self):
        return self._process_composition()

    next = __next__

    def __iter__(self):
        return self


class GlycanHypothesisSerializerBase(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection, hypothesis_name=None):
        DatabaseBoundOperation.__init__(self, database_connection)
        self._hypothesis_name = hypothesis_name
        self._hypothesis_id = None
        self._hypothesis = None
        self.uuid = str(uuid4().hex)

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

    @property
    def hypothesis(self):
        if self._hypothesis is None:
            self._construct_hypothesis()
        return self._hypothesis

    @property
    def hypothesis_name(self):
        if self._hypothesis_name is None:
            self._construct_hypothesis()
        return self._hypothesis_name

    @property
    def hypothesis_id(self):
        if self._hypothesis_id is None:
            self._construct_hypothesis()
        return self._hypothesis_id


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
        self.log("Loading Glycan Compositions from File for %r" % self.hypothesis)
        for composition in self.transformer:
            mass = composition.mass()
            composition_string = composition.serialize()
            formula_string = formula(composition.total_composition())
            inst = DBGlycanComposition(
                calculated_mass=mass, formula=formula_string,
                composition=composition_string,
                hypothesis_id=self.hypothesis_id)
            self.session.add(inst)
        self.session.commit()
