from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)

from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict

from ms_deisotope.output.db import (
    Base, HasUniqueName)


class HypothesisBase(HasUniqueName):
    @declared_attr
    def id(self):
        return Column(Integer, primary_key=True, autoincrement=True)

    parameters = Column(MutableDict.as_mutable(PickleType))


class GlycanHypothesis(HypothesisBase, Base):
    __tablename__ = "GlycanHypothesis"


class GlycopeptideHypothesis(HypothesisBase, Base):
    __tablename__ = "GlycopeptideHypothesis"

    glycan_hypothesis_id = Column(Integer, ForeignKey(GlycanHypothesis.id), index=True)
    glycan_hypothesis = relationship(GlycanHypothesis)
