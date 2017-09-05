from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    Table, Index)

from glycan_profiling.serialize.base import (Base)


from glypy import Composition
from glypy.io import glycoct
from glycopeptidepy import HashableGlycanComposition

from .hypothesis import GlycanHypothesis, GlycopeptideHypothesis


class GlycanBase(object):
    calculated_mass = Column(Numeric(12, 6, asdecimal=False), index=True)
    formula = Column(String(128), index=True)
    composition = Column(String(128))

    def convert(self):
        inst = HashableGlycanComposition.parse(self.composition)
        inst.id = self.id
        return inst

    _glycan_composition = None

    def __getitem__(self, key):
        if self._glycan_composition is None:
            self._glycan_composition = HashableGlycanComposition.parse(self.composition)
        return self._glycan_composition[key]

    def keys(self):
        if self._glycan_composition is None:
            self._glycan_composition = HashableGlycanComposition.parse(self.composition)
        return self._glycan_composition.keys()


class GlycanComposition(GlycanBase, Base):
    __tablename__ = 'GlycanComposition'

    id = Column(Integer, primary_key=True)

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycanHypothesis.id, ondelete="CASCADE"), index=True)

    def __iter__(self):
        return iter(self.keys())

    @declared_attr
    def hypothesis(self):
        return relationship(GlycanHypothesis, backref=backref('glycans', lazy='dynamic'))

    def __repr__(self):
        return "DBGlycanComposition(%s)" % (self.composition)

    structure_classes = relationship("GlycanClass", secondary=lambda: GlycanCompositionToClass, lazy='dynamic')

    __table_args__ = (Index("ix_GlycanComposition_mass_search_index", "calculated_mass", "hypothesis_id"),)


class GlycanStructure(GlycanBase, Base):
    __tablename__ = "GlycanStructure"

    id = Column(Integer, primary_key=True)

    glycan_sequence = Column(String(2048), index=True)

    glycan_composition_id = Column(
        Integer, ForeignKey(GlycanComposition.id, ondelete='CASCADE'), index=True)

    glycan_composition = relationship(GlycanComposition, backref=backref("glycan_structures", lazy='dynamic'))

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycanHypothesis.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def hypothesis(self):
        return relationship(GlycanHypothesis, backref=backref('glycan_structures', lazy='dynamic'))

    structure_classes = relationship("GlycanClass", secondary=lambda: GlycanStructureToClass, lazy='dynamic')

    def convert(self):
        seq = self.glycan_sequence
        structure = glycoct.loads(seq)
        structure.id = self.id
        return structure

    def __repr__(self):
        return "DBGlycanStructure:%d\n%s" % (self.id, self.glycan_sequence)

    __table_args__ = (Index("ix_GlycanStructure_mass_search_index", "calculated_mass", "hypothesis_id"),)


GlycanCombinationGlycanComposition = Table(
    "GlycanCombinationGlycanComposition", Base.metadata,
    Column("glycan_id", Integer, ForeignKey("GlycanComposition.id", ondelete="CASCADE"), index=True),
    Column("combination_id", Integer, ForeignKey("GlycanCombination.id", ondelete="CASCADE"), index=True),
    Column("count", Integer)
)


class GlycanCombination(GlycanBase, Base):
    __tablename__ = 'GlycanCombination'

    id = Column(Integer, primary_key=True)

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycopeptideHypothesis.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def hypothesis(self):
        return relationship(GlycopeptideHypothesis, backref=backref('glycan_combinations', lazy='dynamic'))

    count = Column(Integer)

    components = relationship(
        GlycanComposition,
        secondary=GlycanCombinationGlycanComposition,
        lazy='dynamic')

    def __iter__(self):
        for composition, count in self.components.add_column(
                GlycanCombinationGlycanComposition.c.count):
            i = 0
            while i < count:
                yield composition
                i += 1

    @property
    def component_ids(self):
        session = object_session(self)
        ids = session.query(GlycanCombinationGlycanComposition.c.glycan_id).filter(
            GlycanCombinationGlycanComposition.c.combination_id == self.id).all()
        return [i[0] for i in ids]

    def _get_component_classes(self):
        for case in self:
            yield case.structure_classes.all()

    _component_classes = None

    @property
    def component_classes(self):
        if self._component_classes is None:
            self._component_classes = tuple(self._get_component_classes())
        return self._component_classes

    @hybrid_method
    def dehydrated_mass(self, water_mass=Composition("H2O").mass):
        mass = self.calculated_mass
        return mass - (water_mass * self.count)

    def __repr__(self):
        rep = "GlycanCombination({self.count}, {self.composition})".format(self=self)
        return rep


class GlycanClass(Base):
    __tablename__ = 'GlycanClass'

    id = Column(Integer, primary_key=True)
    name = Column(String(128), index=True)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "GlycanClass(name=%r)" % (self.name,)


class _namespace(object):
    def __repr__(self):
        return "(%r)" % self.__dict__


GlycanTypes = _namespace()
GlycanTypes.n_glycan = "N-Glycan"
GlycanTypes.o_glycan = "O-Glycan"
GlycanTypes.gag_linker = "GAG-Linker"


GlycanCompositionToClass = Table(
    "GlycanCompositionToClass", Base.metadata,
    Column("glycan_id", Integer, ForeignKey("GlycanComposition.id", ondelete="CASCADE"), primary_key=True),
    Column("class_id", Integer, ForeignKey("GlycanClass.id", ondelete="CASCADE"), primary_key=True)
)


GlycanStructureToClass = Table(
    "GlycanStructureToClass", Base.metadata,
    Column("glycan_id", Integer, ForeignKey("GlycanStructure.id", ondelete="CASCADE"), primary_key=True),
    Column("class_id", Integer, ForeignKey("GlycanClass.id", ondelete="CASCADE"), primary_key=True)
)
