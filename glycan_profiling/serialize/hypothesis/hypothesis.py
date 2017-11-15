from collections import Counter

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)

from sqlalchemy.orm import relationship, object_session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict

from glycan_profiling.serialize.base import (
    Base, HasUniqueName)


from glypy.structure.glycan_composition import FrozenGlycanComposition


class HypothesisBase(HasUniqueName):
    @declared_attr
    def id(self):
        return Column(Integer, primary_key=True, autoincrement=True)

    parameters = Column(MutableDict.as_mutable(PickleType), default=dict(), nullable=False)
    status = Column(String(28))

    def __repr__(self):
        return "{self.__class__.__name__}(id={self.id}, name={self.name})".format(self=self)


class GlycanHypothesis(HypothesisBase, Base):
    __tablename__ = "GlycanHypothesis"

    def monosaccharide_bounds(self):
        from .glycan import GlycanComposition
        bounds = Counter()
        session = object_session(self)
        for composition, in session.query(GlycanComposition.composition.distinct()).filter(
                GlycanComposition.hypothesis_id == self.id):
            composition = FrozenGlycanComposition.parse(composition)
            for residue, count in composition.items():
                bounds[residue] = max(bounds[residue], count)
        return {str(k): int(v) for k, v in bounds.items()}


class GlycopeptideHypothesis(HypothesisBase, Base):
    __tablename__ = "GlycopeptideHypothesis"

    glycan_hypothesis_id = Column(Integer, ForeignKey(GlycanHypothesis.id, ondelete="CASCADE"), index=True)
    glycan_hypothesis = relationship(GlycanHypothesis)

    def monosaccharide_bounds(self):
        return self.glycan_hypothesis.monosaccharide_bounds()
