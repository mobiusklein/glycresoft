from collections import Counter

from sqlalchemy import (
    Column, Integer, String, ForeignKey, PickleType, func)

from sqlalchemy.orm import relationship, object_session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict

from glycresoft.serialize.base import (
    Base, HasUniqueName)

from .generic import HasFiles


from glypy.structure.glycan_composition import FrozenGlycanComposition


class HypothesisBase(HasUniqueName, HasFiles):
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

    @property
    def glycan_class_counts(self):
        from .glycan import GlycanClass, GlycanComposition
        session = object_session(self)
        d = {k: v for k, v in session.query(GlycanClass.name, func.count(GlycanClass.name)).join(
             GlycanComposition.structure_classes).group_by(GlycanClass.name).all()}
        return d

    @property
    def n_glycan_only(self):
        counts = self.glycan_class_counts
        k = counts.get('N-Glycan', 0)
        return len(counts) == 1 and k > 0


class GlycopeptideHypothesis(HypothesisBase, Base):
    __tablename__ = "GlycopeptideHypothesis"

    glycan_hypothesis_id = Column(Integer, ForeignKey(GlycanHypothesis.id, ondelete="CASCADE"), index=True)
    glycan_hypothesis = relationship(GlycanHypothesis)

    def monosaccharide_bounds(self):
        return self.glycan_hypothesis.monosaccharide_bounds()

    @property
    def glycan_class_counts(self):
        return self.glycan_hypothesis.glycan_class_counts

    @property
    def n_glycan_only(self):
        return self.glycan_hypothesis.n_glycan_only