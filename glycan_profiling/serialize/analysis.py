from functools import partial

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table, func)

from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict

import dill

from .base import (
    Base, HasUniqueName, SampleRun)

from glypy.utils import Enum


DillType = partial(PickleType, pickler=dill)


class AnalysisTypeEnum(Enum):
    glycopeptide_lc_msms = 0
    glycan_lc_ms = 1


AnalysisTypeEnum.glycopeptide_lc_msms.add_name("Glycopeptide LC-MS/MS")
AnalysisTypeEnum.glycan_lc_ms.add_name("Glycan LC-MS")


class Analysis(Base, HasUniqueName):
    __tablename__ = "Analysis"

    id = Column(Integer, primary_key=True)
    sample_run_id = Column(Integer, ForeignKey(SampleRun.id, ondelete="CASCADE"), index=True)
    sample_run = relationship(SampleRun)
    analysis_type = Column(String(128))
    parameters = Column(MutableDict.as_mutable(DillType))
    status = Column(String(28))

    def __repr__(self):
        sample_run = self.sample_run
        if sample_run:
            sample_run_name = sample_run.name
        else:
            sample_run_name = "<Detached From Sample>"
        return "Analysis(%s, %s)" % (self.name, sample_run_name)

    def _infer_hypothesis_id(self):
        try:
            hypothesis_id = self.parameters['hypothesis_id']
            return hypothesis_id
        except KeyError:
            session = object_session(self)
            if self.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
                from . import IdentifiedGlycopeptide, GlycopeptideHypothesis, Glycopeptide
                hypothesis_id = session.query(func.distinct(GlycopeptideHypothesis.id)).join(
                    Glycopeptide).join(
                    IdentifiedGlycopeptide,
                    Glycopeptide.id == IdentifiedGlycopeptide.structure_id).filter(
                    IdentifiedGlycopeptide.analysis_id == self.id).scalar()
                return hypothesis_id
            elif self.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
                from . import GlycanComposition, GlycanCompositionChromatogram, GlycanHypothesis
                hypothesis_id = session.query(func.distinct(GlycanHypothesis.id)).join(GlycanComposition).join(
                    GlycanCompositionChromatogram,
                    GlycanCompositionChromatogram.glycan_composition_id == GlycanComposition.id).filter(
                    GlycanCompositionChromatogram.analysis_id == self.id).scalar()
                return hypothesis_id
            else:
                raise ValueError(self.analysis_type)

    @property
    def hypothesis_id(self):
        return self._infer_hypothesis_id()

    def get_hypothesis(self):
        from . import GlycopeptideHypothesis, GlycanHypothesis
        hypothesis_id = self.hypothesis_id
        if hypothesis_id is None:
            raise ValueError(
                "Analysis is does not have a Hypothesis (analysis id: %r, name: %r)" % (
                    self.id, self.name))
        session = object_session(self)
        if self.analysis_type == AnalysisTypeEnum.glycopeptide_lc_msms:
            return session.query(GlycopeptideHypothesis).get(hypothesis_id)
        elif self.analysis_type == AnalysisTypeEnum.glycan_lc_ms:
            return session.query(GlycanHypothesis).get(hypothesis_id)

    hypothesis = property(get_hypothesis)


class BoundToAnalysis(object):

    @declared_attr
    def analysis_id(self):
        return Column(Integer, ForeignKey(Analysis.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def analysis(self):
        return relationship(Analysis)
