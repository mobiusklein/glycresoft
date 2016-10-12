from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.orm import relationship, backref
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    Boolean, Table)

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

    @property
    def __getitem__(self, key):
        if self._glycan_composition is None:
            self._glycan_composition = FrozenGlycanComposition.parse(self.composition)
        return self._glycan_composition[key]


class GlycanComposition(GlycanBase, Base):
    __tablename__ = 'GlycanComposition'

    id = Column(Integer, primary_key=True)

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycanHypothesis.id, ondelete="CASCADE"), index=True)

    def __repr__(self):
        return "DBGlycanComposition(%s)" % (self.composition)


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


class MonosaccharideResidue(Base):
    __tablename__ = "MonosaccharideResidue"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)


MonosaccharideResidueCountToGlycanComposition = Table(
    "MonosaccharideResidueCountToGlycanComposition", Base.metadata,
    Column("glycan_id", Integer, ForeignKey(GlycanComposition.id, ondelete='CASCADE'), primary_key=True),
    Column("monosaccharide_id", Integer, ForeignKey(MonosaccharideResidue.id, ondelete='CASCADE'), primary_key=True),
    Column("count", Integer))


MonosaccharideResidueCountToGlycanCombination = Table(
    "MonosaccharideResidueCountToGlycanCombination", Base.metadata,
    Column("glycan_id", Integer, ForeignKey(GlycanCombination.id, ondelete='CASCADE'), primary_key=True),
    Column("monosaccharide_id", Integer, ForeignKey(MonosaccharideResidue.id, ondelete='CASCADE'), primary_key=True),
    Column("count", Integer))
