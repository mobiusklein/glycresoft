from weakref import WeakValueDictionary

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)
from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy.ext.declarative import declared_attr

from glycan_profiling.tandem.spectrum_matcher_base import (
    SpectrumMatch as MemorySpectrumMatch, SpectrumSolutionSet as MemorySpectrumSolutionSet, SpectrumReference,
    TargetReference)

from .analysis import BoundToAnalysis
from .hypothesis import Glycopeptide

from ms_deisotope.output.db import (
    Base, MSScan)


class SpectrumMatchBase(BoundToAnalysis):

    score = Column(Numeric(12, 6, asdecimal=False), index=True)

    @declared_attr
    def scan_id(self):
        return Column(Integer, ForeignKey(MSScan.id), index=True)

    @declared_attr
    def scan(self):
        return relationship(MSScan)


class GlycopeptideSpectrumCluster(Base, BoundToAnalysis):
    __tablename__ = "GlycopeptideSpectrumCluster"

    id = Column(Integer, primary_key=True)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, analysis_id, *args, **kwargs):
        inst = cls()
        session.add(inst)
        session.flush()
        cluster_id = inst.id
        for solution_set in obj.tandem_solutions:
            GlycopeptideSpectrumSolutionSet.serialize(
                solution_set, session, scan_look_up_cache, analysis_id,
                cluster_id, *args, **kwargs)
        return inst

    def convert(self):
        return [x.convert() for x in self.spectrum_solutions]


class GlycopeptideSpectrumSolutionSet(Base, BoundToAnalysis):
    __tablename__ = "GlycopeptideSpectrumSolutionSet"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(
        Integer,
        ForeignKey(GlycopeptideSpectrumCluster.id, ondelete="CASCADE"),
        index=True)

    cluster = relationship(GlycopeptideSpectrumCluster, backref=backref("spectrum_solutions", lazy='subquery'))

    is_decoy = Column(Boolean, index=True)

    scan_id = Column(Integer, ForeignKey(MSScan.id), index=True)
    scan = relationship(MSScan)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, analysis_id, cluster_id, is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            cluster_id=cluster_id)
        session.add(inst)
        session.flush()
        for solution in obj:
            GlycopeptideSpectrumMatch.serialize(
                solution, session, scan_look_up_cache,
                analysis_id, inst.id, is_decoy, *args, **kwargs)
        return inst

    def convert(self):
        matches = [x.convert() for x in self.spectrum_matches]
        matches.sort(key=lambda x: x.score, reverse=True)
        inst = MemorySpectrumSolutionSet(
            SpectrumReference(self.scan.scan_id, self.scan.precursor_information),
            matches
        )
        inst.q_value = min(x.q_value for x in inst)
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


class GlycopeptideSpectrumMatch(Base, SpectrumMatchBase):
    __tablename__ = "GlycopeptideSpectrumMatch"

    id = Column(Integer, primary_key=True)
    solution_set_id = Column(
        Integer, ForeignKey(
            GlycopeptideSpectrumSolutionSet.id, ondelete='CASCADE'),
        index=True)
    solution_set = relationship(GlycopeptideSpectrumSolutionSet, backref=backref("spectrum_matches", lazy='subquery'))
    q_value = Column(Numeric(8, 7, asdecimal=False), index=True)
    is_decoy = Column(Boolean, index=True)
    is_best_match = Column(Boolean, index=True)

    structure_id = Column(
        Integer, ForeignKey(Glycopeptide.id, ondelete='CASCADE'),
        index=True)

    structure = relationship(Glycopeptide)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, analysis_id, solution_set_id, is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            score=obj.score,
            q_value=obj.q_value,
            solution_set_id=solution_set_id,
            is_best_match=obj.best_match,
            structure_id=obj.target.id)
        session.add(inst)
        session.flush()
        return inst

    def convert(self):
        session = object_session(self)
        scan = session.query(MSScan).get(self.scan_id).convert()
        target = session.query(Glycopeptide).get(self.structure_id).convert()
        inst = MemorySpectrumMatch(scan, target, self.score, self.is_best_match)
        inst.q_value = self.q_value
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())
