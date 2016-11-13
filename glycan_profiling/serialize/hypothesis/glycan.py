from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.orm import relationship, backref, aliased, object_session
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    Boolean, Table, PrimaryKeyConstraint)

from ms_deisotope.output.db import (Base)


from glypy import Composition
from glypy.composition.glycan_composition import FrozenGlycanComposition
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

    @hybrid_method
    def dehydrated_mass(self, water_mass=Composition("H2O").mass):
        mass = self.calculated_mass
        return mass - (water_mass * self.count)

    def __repr__(self):
        rep = "GlycanCombination({self.count}, {self.composition})".format(self=self)
        return rep


GlycanCompositionGraphEdge = Table(
    "GlycanCompositionGraphEdge", Base.metadata,
    Column("source_id", Integer, ForeignKey(GlycanComposition.id, ondelete="CASCADE"), primary_key=True),
    Column("terminal_id", Integer, ForeignKey(GlycanComposition.id, ondelete="CASCADE"), primary_key=True),
    Column("order", Numeric(6, 3, asdecimal=False)),
    Column('weight', Numeric(8, 4, asdecimal=False)))


class GlycanClass(Base):
    __tablename__ = 'GlycanClass'

    id = Column(Integer, primary_key=True)
    name = Column(String(128), index=True)


GlycanCompositionToClass = Table(
    "GlycanCompositionToClass", Base.metadata,
    Column("glycan_id", Integer, ForeignKey("GlycanComposition.id", ondelete="CASCADE"), index=True),
    Column("class_id", Integer, ForeignKey("GlycanClass.id", ondelete="CASCADE"), index=True)
)
