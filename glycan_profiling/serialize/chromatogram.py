from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.declarative import declared_attr

from glycan_profiling.chromatogram_tree import (
    ChromatogramTreeNode as MemoryChromatogramTreeNode, ChromatogramTreeList, Chromatogram as MemoryChromatogram,
    MassShift as MemoryMassShift, CompoundMassShift as MemoryCompoundMassShift)

from glycan_profiling.scoring import (
    ChromatogramSolution as MemoryChromatogramSolution)

from .analysis import BoundToAnalysis

from ms_deisotope.output.db import (
    Base, DeconvolutedPeak, MSScan, Mass)

from glypy.composition.base import formula
from glypy import Composition


def extract_key(obj):
    try:
        return obj._key()
    except AttributeError:
        if obj is None:
            return obj
        return str(obj)


class SimpleSerializerCacheBase(object):
    _model_class = None

    def __init__(self, session, store=None):
        if store is None:
            store = dict()
        self.session = session
        self.store = store

    def serialize(self, obj, *args, **kwargs):
        if obj is None:
            return None
        try:
            db_obj = self.store[extract_key(obj)]
            return db_obj
        except KeyError:
            db_obj = self._model_class.serialize(obj, self.session, *args, **kwargs)
            self.store[extract_key(db_obj)] = db_obj
            return db_obj


class MassShift(Base):
    __tablename__ = "MassShift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), index=True, unique=True)
    composition = Column(String(128))

    def convert(self):
        return MemoryMassShift(str(self.name), Composition(str(self.composition)))

    @classmethod
    def serialize(cls, obj, session, *args, **kwargs):
        shift = session.query(MassShift).filter(MassShift.name == obj.name).all()
        if shift:
            return shift[0]
        else:
            db_obj = MassShift(name=obj.name, composition=formula(obj.composition))
            session.add(db_obj)
            session.flush()
            return db_obj

    def __repr__(self):
        return "DB" + repr(self.convert())

    def _key(self):
        return self.name


class CompoundMassShift(Base):
    __tablename__ = "CompoundMassShift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), index=True, unique=True)

    counts = association_proxy(
        "_counts", "count", creator=lambda k, v: MassShiftToCompoundMassShift(individual_id=k.id, count=v))

    def convert(self):
        return MemoryCompoundMassShift({k.convert(): v for k, v in self.counts.items()})

    def __repr__(self):
        return "DB" + repr(self.convert())

    @classmethod
    def serialize(cls, obj, session, *args, **kwargs):
        shift = session.query(CompoundMassShift).filter(CompoundMassShift.name == obj.name).all()
        if shift:
            return shift[0]
        else:
            db_obj = CompoundMassShift(name=obj.name)
            session.add(db_obj)
            session.flush()
            if isinstance(obj, MemoryMassShift):
                db_member = MassShift.serialize(obj, session, *args, **kwargs)
                db_obj.counts[db_member] = 1
            else:
                for member, count in obj.counts.items():
                    db_member = MassShift.serialize(member, session, *args, **kwargs)
                    db_obj.counts[db_member] = count
            session.add(db_obj)
            session.flush()
            return db_obj

    def _key(self):
        return self.name


class MassShiftToCompoundMassShift(Base):
    __tablename__ = "MassShiftToCompoundMassShift"

    compound_id = Column(Integer, ForeignKey(CompoundMassShift.id, ondelete="CASCADE"), index=True, primary_key=True)
    individual_id = Column(Integer, ForeignKey(MassShift.id, ondelete='CASCADE'), index=True, primary_key=True)
    individual = relationship(MassShift)
    compound = relationship(CompoundMassShift, backref=backref(
        "_counts", collection_class=attribute_mapped_collection("individual.name"),
        cascade="all, delete-orphan"))

    count = Column(Integer)


class MassShiftSerializer(SimpleSerializerCacheBase):
    _model_class = CompoundMassShift


class ChromatogramTreeNode(Base):
    __tablename__ = "ChromatogramTreeNode"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_type_name = Column(String(64), ForeignKey(CompoundMassShift.name), index=True)
    scan_id = Column(String(128), ForeignKey(MSScan.scan_id), index=True)
    retention_time = Column(Numeric(8, 4, asdecimal=False), index=True)
    neutral_mass = Mass()

    node_type = relationship(CompoundMassShift)
    children = association_proxy("_children", "child", creator=lambda x: ChromatogramTreeNodeBranch(child_id=x.id))

    members = relationship(DeconvolutedPeak, secondary=lambda: ChromatogramTreeNodeToDeconvolutedPeak)

    def convert(self):
        inst = MemoryChromatogramTreeNode(
            self.retention_time, self.scan_id, [
                child.convert() for child in self.children],
            [p.convert() for p in self.members],
            self.node_type.convert())
        return inst

    @classmethod
    def serialize(cls, obj, session, peak_lookup_table=None, mass_shift_cache=None, *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = MassShiftSerializer(session)
        inst = ChromatogramTreeNode(
            scan_id=obj.scan_id, retention_time=obj.retention_time)
        nt = mass_shift_cache.serialize(obj.node_type)
        inst.node_type_name = nt.name
        session.add(inst)
        session.flush()
        if peak_lookup_table is not None:
            member_ids = []
            for member in obj.members:
                member_ids.append(peak_lookup_table[obj.scan_id, member])
            session.execute(ChromatogramTreeNodeToDeconvolutedPeak.insert(), [
                {'node_id': inst.id, 'peak_id': member_id} for member_id in member_ids])
        children = [cls.serialize(
            child, session, peak_lookup_table=peak_lookup_table,
            mass_shift_cache=mass_shift_cache,
            *args, **kwargs) for child in obj.children]
        inst.children.extend(children)

        session.add(inst)
        session.flush()
        return inst


class ChromatogramTreeNodeBranch(Base):
    __tablename__ = "ChromatogramTreeNodeBranch"

    parent_id = Column(Integer, ForeignKey(ChromatogramTreeNode.id, ondelete="CASCADE"), index=True, primary_key=True)
    child_id = Column(Integer, ForeignKey(ChromatogramTreeNode.id, ondelete="CASCADE"), index=True, primary_key=True)
    child = relationship(ChromatogramTreeNode, backref=backref(
        "_children"), foreign_keys=[child_id])
    parent = relationship(ChromatogramTreeNode, backref=backref(
        "parent"),
        foreign_keys=[parent_id])


ChromatogramTreeNodeToDeconvolutedPeak = Table(
    "ChromatogramTreeNodeToDeconvolutedPeak", Base.metadata,
    Column("node_id", Integer, ForeignKey(
        ChromatogramTreeNode.id, ondelete='CASCADE'), primary_key=True),
    Column("peak_id", Integer, ForeignKey(
        DeconvolutedPeak.id, ondelete="CASCADE"), primary_key=True))


class Chromatogram(Base, BoundToAnalysis):
    __tablename__ = "Chromatogram"

    id = Column(Integer, primary_key=True, autoincrement=True)
    neutral_mass = Mass()

    @property
    def composition(self):
        return None

    start_time = Column(Numeric(8, 4, asdecimal=False), index=True)
    end_time = Column(Numeric(8, 4, asdecimal=False), index=True)

    nodes = relationship(ChromatogramTreeNode, secondary=lambda: ChromatogramToChromatogramTreeNode)

    def convert(self):
        nodes = [node.convert() for node in self.nodes]
        nodes.sort(key=lambda x: x.retention_time)
        inst = MemoryChromatogram(None, ChromatogramTreeList(nodes))
        return inst

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None, *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = MassShiftSerializer(session)
        inst = cls(
            neutral_mass=obj.neutral_mass, start_time=obj.start_time, end_time=obj.end_time,
            analysis_id=analysis_id)
        session.add(inst)
        session.flush()
        node_ids = []
        for node in obj.nodes:
            db_node = ChromatogramTreeNode.serialize(
                node, session, peak_lookup_table, mass_shift_cache,
                *args, **kwargs)
            node_ids.append(db_node.id)
        session.execute(ChromatogramToChromatogramTreeNode.insert(), [
            {"chromatogram_id": inst.id, "node_id": node_id} for node_id in node_ids])
        return inst

    def _total_signal_query(session):
        return session.query(
            Chromatogram,
            func.sum(DeconvolutedPeak.intensity)).join(
            Chromatogram.nodes).join(ChromatogramTreeNode.members).group_by(
            ChromatogramSolution.id)

    def __repr__(self):
        return "DB" + repr(self.convert())


ChromatogramToChromatogramTreeNode = Table(
    "ChromatogramToChromatogramTreeNode", Base.metadata,
    Column("chromatogram_id", Integer, ForeignKey(
        Chromatogram.id, ondelete="CASCADE"), primary_key=True),
    Column("node_id", Integer, ForeignKey(
        ChromatogramTreeNode.id, ondelete='CASCADE'), primary_key=True))


class CompositionGroup(Base, BoundToAnalysis):
    __tablename__ = "CompositionGroup"

    id = Column(Integer, primary_key=True)
    composition = Column(String(128), index=True)

    @classmethod
    def serialize(cls, obj, session, *args, **kwargs):
        if obj is None:
            return None
        composition = session.query(cls).filter(cls.composition == str(obj)).all()
        if composition:
            return composition[0]
        else:
            db_obj = cls(composition=str(obj))
            session.add(db_obj)
            session.flush()
            return db_obj

    def convert(self):
        return str(self.composition)

    def _key(self):
        return self.composition

    def __repr__(self):
        return "CompositionGroup(%r, %d)" % (repr(self.convert()), self.id)


class CompositionGroupSerializer(SimpleSerializerCacheBase):
    _model_class = CompositionGroup


class ChromatogramSolution(Base, BoundToAnalysis):
    __tablename__ = "ChromatogramSolution"

    id = Column(Integer, primary_key=True)

    score = Column(Numeric(8, 7, asdecimal=False), index=True)
    chromatogram_id = Column(Integer, ForeignKey(
        Chromatogram.id, ondelete='CASCADE'), index=True)
    composition_group_id = Column(Integer, ForeignKey(
        CompositionGroup.id, ondelete='CASCADE'), index=True)

    chromatogram = relationship(Chromatogram)
    composition_group = relationship(CompositionGroup)

    @property
    def neutral_mass(self):
        return self.chromatogram.neutral_mass

    @property
    def start_time(self):
        return self.chromatogram.start_time

    @property
    def end_time(self):
        return self.chromatogram.end_time

    @property
    def composition(self):
        return self.composition_group

    def convert(self):
        chromatogram = self.chromatogram.convert()
        composition = self.composition_group.convert() if self.composition_group else None
        chromatogram.composition = composition
        sol = MemoryChromatogramSolution(chromatogram, self.score)
        return sol

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  composition_cache=None, *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = MassShiftSerializer(session)
        if composition_cache is None:
            composition_cache = CompositionGroupSerializer(session)
        db_composition_group = composition_cache.serialize(obj.composition)
        composition_group_id = db_composition_group.id if db_composition_group is not None else None
        inst = cls(
            composition_group_id=composition_group_id, analysis_id=analysis_id,
            score=obj.score)
        chromatogram = Chromatogram.serialize(
            obj.chromatogram, session=session, analysis_id=analysis_id, peak_lookup_table=peak_lookup_table,
            mass_shift_cache=mass_shift_cache, *args, **kwargs)
        inst.chromatogram_id = chromatogram.id
        session.add(inst)
        session.flush()
        return inst

    def _total_signal_query(session):
        return session.query(
            ChromatogramSolution,
            func.sum(DeconvolutedPeak.intensity)).join(
            ChromatogramSolution.chromatogram).join(
            Chromatogram.nodes).join(ChromatogramTreeNode.members).group_by(
            ChromatogramSolution.id)

    def __repr__(self):
        return "DB" + repr(self.convert())


ChromatogramSolutionAdductedToCompositionGroup = Table(
    "ChromatogramSolutionAdductedToCompositionGroup", Base.metadata,
    Column("adducted_solution_id", Integer, ForeignKey(ChromatogramSolution.id, ondelete="CASCADE"), primary_key=True),
    Column("mass_shift_id", Integer, ForeignKey(CompoundMassShift.id, ondelete='CASCADE'), primary_key=True),
    Column("owning_composition_id", Integer, ForeignKey(CompositionGroup.id, ondelete="CASCADE"), primary_key=True))
