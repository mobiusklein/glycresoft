import re
from collections import OrderedDict

from sqlalchemy.ext.baked import bakery
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.orm import relationship, backref, make_transient, Query, validates
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table, Text)
from sqlalchemy.ext.mutable import MutableDict

from ms_deisotope.output.db import (
    Base, MutableList)


from glypy import Composition
from glypy.composition.glycan_composition import FrozenGlycanComposition

from .hypothesis import GlycanHypothesis


class GlycanBase(object):

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycanHypothesis.id, ondelete="CASCADE"), index=True)

    calculated_mass = Column(Numeric(12, 6, asdecimal=False), index=True)
    formula = Column(String(128), index=True)
    composition = Column(String(128))

    def convert(self):
        return FrozenGlycanComposition.parse(self.composition)


class GlycanComposition(GlycanBase, Base):
    __tablename__ = 'GlycanComposition'

    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return "DBGlycanComposition(%s)" % (self.composition)


GlycanCombinationGlycanComposition = Table(
    "GlycanCombinationGlycanComposition", Base.metadata,
    Column("glycan_id", Integer, ForeignKey("GlycanComposition.id"), index=True),
    Column("combination_id", Integer, ForeignKey("GlycanCombination.id"), index=True),
    Column("count", Integer)
)


class GlycanCombination(GlycanBase, Base):
    __tablename__ = 'GlycanCombination'

    id = Column(Integer, primary_key=True)
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

    @hybrid_method
    def dehydrated_mass(self, water_mass=Composition("H2O").mass):
        mass = self.calculated_mass
        return mass - (water_mass * self.count)

    def __repr__(self):
        rep = "GlycanCombination({self.count}, {self.composition})".format(self=self)
        return rep
