from uuid import uuid4

from glycan_profiling.serialize import (
    GlycanHypothesis, GlycanComposition,
    DatabaseBoundOperation, GlycanCompositionToClass)

from glycan_profiling.serialize.utils import get_uri_for_instance

from .glycan_source import GlycanClassLoader

from glycan_profiling.task import TaskBase


class GlycanHypothesisMigrator(DatabaseBoundOperation, TaskBase):
    def __init__(self, database_connection):
        DatabaseBoundOperation.__init__(self, database_connection)
        self.hypothesis_id = None
        self.glycan_composition_id_map = dict()
        self._structure_class_loader = None

    def make_structure_class_loader(self):
        return GlycanClassLoader(self.session)

    @property
    def structure_class_loader(self):
        if self._structure_class_loader is None:
            self._structure_class_loader = self.make_structure_class_loader()
        return self._structure_class_loader

    def migrate_hypothesis(self, hypothesis):
        new_hypothesis = GlycanHypothesis(
            name=hypothesis.name,
            uuid=str(uuid4().hex),
            status=hypothesis.status,
            parameters=dict(hypothesis.parameters or {}))
        new_hypothesis.parameters['original_uuid'] = hypothesis.uuid
        new_hypothesis.parameters['copied_from'] = get_uri_for_instance(hypothesis)
        self.session.add(new_hypothesis)
        self.session.flush()
        self.hypothesis_id = new_hypothesis.id

    def _migrate(self, obj):
        self.session.add(obj)
        self.session.flush()
        new_id = obj.id
        return new_id

    def commit(self):
        self.session.commit()

    def migrate_glycan_composition(self, glycan_composition):
        gc = glycan_composition
        new_gc = GlycanComposition(
            calculated_mass=gc.calculated_mass,
            formula=gc.formula,
            composition=gc.composition,
            hypothesis_id=self.hypothesis_id)
        new_id = self._migrate(new_gc)
        self.glycan_composition_id_map[gc.id] = new_id

        class_types = []
        for glycan_class in gc.structure_classes:
            local_glycan_class = self.structure_class_loader[glycan_class.name]
            class_types.append({"glycan_id": new_id, "class_id": local_glycan_class.id})
        if class_types:
            self.session.execute(GlycanCompositionToClass.insert(), class_types)
