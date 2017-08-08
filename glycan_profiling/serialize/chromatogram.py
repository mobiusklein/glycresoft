from collections import Counter

import numpy as np

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table, func, select, join, alias)
from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.declarative import declared_attr

from glycan_profiling.chromatogram_tree import (
    ChromatogramTreeNode as MemoryChromatogramTreeNode,
    ChromatogramTreeList,
    Chromatogram as MemoryChromatogram,
    MassShift as MemoryMassShift,
    CompoundMassShift as MemoryCompoundMassShift,
    GlycanCompositionChromatogram as MemoryGlycanCompositionChromatogram,
    ChromatogramInterface)

from glycan_profiling.chromatogram_tree.chromatogram import MIN_POINTS_FOR_CHARGE_STATE

from glycan_profiling.scoring import (
    ChromatogramSolution as MemoryChromatogramSolution)
from glycan_profiling.models import GeneralScorer

from .analysis import BoundToAnalysis
from .hypothesis import GlycanComposition

from ms_deisotope.output.db import (
    Base, DeconvolutedPeak, MSScan, Mass, make_memory_deconvoluted_peak)

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

    def __eq__(self, other):
        return self.name == extract_key(other)

    def __hash__(self):
        return hash(self.name)


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
        "_counts", collection_class=attribute_mapped_collection("individual"),
        cascade="all, delete-orphan"))

    count = Column(Integer)


class MassShiftSerializer(SimpleSerializerCacheBase):
    _model_class = CompoundMassShift


def _create_chromatogram_tree_node_branch(x):
    return ChromatogramTreeNodeBranch(child_id=x.id)


class ChromatogramTreeNode(Base, BoundToAnalysis):
    __tablename__ = "ChromatogramTreeNode"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_type_id = Column(Integer, ForeignKey(CompoundMassShift.id, ondelete='CASCADE'), index=True)
    scan_id = Column(Integer, ForeignKey(MSScan.id, ondelete='CASCADE'), index=True)
    retention_time = Column(Numeric(8, 4, asdecimal=False), index=True)
    neutral_mass = Mass()

    scan = relationship(MSScan)
    node_type = relationship(CompoundMassShift, lazy='joined')

    children = association_proxy("_children", "child", creator=_create_chromatogram_tree_node_branch)

    members = relationship(DeconvolutedPeak, secondary=lambda: ChromatogramTreeNodeToDeconvolutedPeak, lazy="subquery")

    def charge_states(self):
        u = set()
        u.update({p.charge for p in self.members})
        for child in self.children:
            u.update(child.charge_states())
        return u

    @classmethod
    def _convert(cls, session, id, node_type_cache=None, scan_id_cache=None):
        if node_type_cache is None:
            node_type_cache = dict()
        if scan_id_cache is None:
            scan_id_cache = dict()

        node_table = cls.__table__
        attribs = session.execute(node_table.select().where(node_table.c.id == id)).fetchone()

        peak_table = DeconvolutedPeak.__table__

        selector = select([DeconvolutedPeak.__table__]).select_from(
            join(peak_table, ChromatogramTreeNodeToDeconvolutedPeak)).where(
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == id)

        members = session.execute(selector).fetchall()

        try:
            scan_id = scan_id_cache[attribs.scan_id]
        except KeyError:
            selector = select([MSScan.__table__.c.scan_id]).where(MSScan.__table__.c.id == attribs.scan_id)
            scan_id = session.execute(selector).fetchone()
            scan_id_cache[attribs.scan_id] = scan_id

        members = [make_memory_deconvoluted_peak(m) for m in members]

        try:
            node_type = node_type_cache[attribs.node_type_id]
        except KeyError:
            shift = session.query(CompoundMassShift).get(attribs.node_type_id)
            node_type = shift.convert()
            node_type_cache[attribs.node_type_id] = node_type

        children_ids = session.query(ChromatogramTreeNodeBranch.child_id).filter(
            ChromatogramTreeNodeBranch.parent_id == id)
        children = [cls._convert(session, i[0], node_type_cache) for i in children_ids]

        return MemoryChromatogramTreeNode(
            attribs.retention_time, scan_id, children, members,
            node_type)

    def convert(self, node_type_cache=None, scan_id_cache=None):
        inst = MemoryChromatogramTreeNode(
            self.retention_time, self.scan.scan_id, [
                child.convert() for child in self.children],
            [p.convert() for p in self.members],
            self.node_type.convert())
        return inst

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None, scan_lookup_table=None,
                  node_peak_map=None, *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = MassShiftSerializer(session)
        inst = ChromatogramTreeNode(
            scan_id=scan_lookup_table[obj.scan_id], retention_time=obj.retention_time,
            analysis_id=analysis_id)
        nt = mass_shift_cache.serialize(obj.node_type)
        inst.node_type_id = nt.id

        session.add(inst)
        session.flush()

        if peak_lookup_table is not None:
            member_ids = []
            blocked = 0
            for member in obj.members:
                peak_id = peak_lookup_table[obj.scan_id, member]
                node_peak_key = (inst.id, peak_id)
                if node_peak_key in node_peak_map:
                    blocked += 1
                    continue
                node_peak_map[node_peak_key] = True
                member_ids.append(peak_id)

            if len(member_ids):
                session.execute(ChromatogramTreeNodeToDeconvolutedPeak.insert(), [
                    {'node_id': inst.id, 'peak_id': member_id} for member_id in member_ids])
            elif blocked == 0:
                raise Exception("No Peaks Saved")

        children = [cls.serialize(
            child, session,
            analysis_id=analysis_id,
            peak_lookup_table=peak_lookup_table,
            mass_shift_cache=mass_shift_cache,
            scan_lookup_table=scan_lookup_table,
            node_peak_map=node_peak_map,
            *args, **kwargs) for child in obj.children]
        branches = [
            ChromatogramTreeNodeBranch(parent_id=inst.id, child_id=child.id)
            for child in children
        ]
        assert len(branches) == len(obj.children)
        session.add_all(branches)
        session.add(inst)
        # assert len(inst.children) == len(obj.children)
        session.flush()
        return inst

    def _total_intensity_query(self, session):
        session.query()


class ChromatogramTreeNodeBranch(Base):
    __tablename__ = "ChromatogramTreeNodeBranch"

    parent_id = Column(Integer, ForeignKey(ChromatogramTreeNode.id, ondelete="CASCADE"), index=True, primary_key=True)
    child_id = Column(Integer, ForeignKey(ChromatogramTreeNode.id, ondelete="CASCADE"), index=True, primary_key=True)
    child = relationship(ChromatogramTreeNode, backref=backref(
        "parent"), foreign_keys=[child_id])
    parent = relationship(ChromatogramTreeNode, backref=backref(
        "_children", lazy='subquery'),
        foreign_keys=[parent_id])

    def __repr__(self):
        return "ChromatogramTreeNodeBranch(%r, %r)" % (self.parent_id, self.child_id)


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

    @property
    def adducts(self):
        session = object_session(self)
        return self._adducts_query(session)

    def _adducts_query(self, session):
        anode = alias(ChromatogramTreeNode.__table__)
        bnode = alias(ChromatogramTreeNode.__table__)
        apeak = alias(DeconvolutedPeak.__table__)

        peak_join = apeak.join(
            ChromatogramTreeNodeToDeconvolutedPeak,
            ChromatogramTreeNodeToDeconvolutedPeak.c.peak_id == apeak.c.id)

        root_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == anode.c.id)

        branch_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramTreeNodeBranch,
            ChromatogramTreeNodeBranch.child_id == anode.c.id).join(
            bnode, ChromatogramTreeNodeBranch.parent_id == bnode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == bnode.c.id)

        branch_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        root_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        all_node_info_q = root_node_info.union_all(branch_node_info).order_by(
            anode.c.retention_time)

        all_node_info = session.execute(all_node_info_q).fetchall()

        node_type_ids = set()
        for node_type_id, rt in all_node_info:
            node_type_ids.add(node_type_id)

        node_types = []
        for ntid in node_type_ids:
            node_types.append(session.query(CompoundMassShift).get(ntid).convert())
        return node_types

    def get_chromatogram(self):
        # return self.convert()
        return self

    @property
    def charge_states(self):
        states = Counter()
        for node in self.nodes:
            states += (Counter(node.charge_states()))
        # Require more than `MIN_POINTS_FOR_CHARGE_STATE` data points to accept any
        # charge state
        collapsed_states = {k for k, v in states.items() if v >= min(MIN_POINTS_FOR_CHARGE_STATE, len(self))}
        if not collapsed_states:
            collapsed_states = set(states.keys())
        return collapsed_states

    def raw_convert(self, node_type_cache=None, scan_id_cache=None):
        session = object_session(self)
        node_ids = session.query(ChromatogramToChromatogramTreeNode.c.node_id).filter(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id).all()
        nodes = [ChromatogramTreeNode._convert(
            session, ni[0], node_type_cache=node_type_cache,
            scan_id_cache=scan_id_cache) for ni in node_ids]
        nodes.sort(key=lambda x: x.retention_time)
        inst = MemoryChromatogram(None, ChromatogramTreeList(nodes))
        return inst

    def orm_convert(self, *args, **kwargs):
        nodes = [node.convert(*args, **kwargs) for node in self.nodes]
        nodes.sort(key=lambda x: x.retention_time)
        inst = MemoryChromatogram(None, ChromatogramTreeList(nodes))
        return inst

    def convert(self, node_type_cache=None, scan_id_cache=None):
        return self.orm_convert(node_type_cache, scan_id_cache)

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, node_peak_map=None, *args, **kwargs):
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
                node, session, analysis_id=analysis_id,
                peak_lookup_table=peak_lookup_table,
                mass_shift_cache=mass_shift_cache,
                scan_lookup_table=scan_lookup_table,
                node_peak_map=node_peak_map, *args, **kwargs)
            node_ids.append(db_node.id)
        session.execute(ChromatogramToChromatogramTreeNode.insert(), [
            {"chromatogram_id": inst.id, "node_id": node_id} for node_id in node_ids])
        return inst

    def _weighted_neutral_mass_query(self, session):
        anode = alias(ChromatogramTreeNode.__table__)
        bnode = alias(ChromatogramTreeNode.__table__)
        apeak = alias(DeconvolutedPeak.__table__)

        peak_join = apeak.join(
            ChromatogramTreeNodeToDeconvolutedPeak,
            ChromatogramTreeNodeToDeconvolutedPeak.c.peak_id == apeak.c.id)

        root_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == anode.c.id)

        branch_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramTreeNodeBranch,
            ChromatogramTreeNodeBranch.child_id == anode.c.id).join(
            bnode, ChromatogramTreeNodeBranch.parent_id == bnode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == bnode.c.id)

        branch_intensities = select([apeak.c.intensity, apeak.c.neutral_mass]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        root_intensities = select([apeak.c.intensity, apeak.c.neutral_mass]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        all_intensity_mass_q = root_intensities.union_all(branch_intensities)

        all_intensity_mass = session.execute(all_intensity_mass_q).fetchall()

        arr = np.array(all_intensity_mass)
        intensity = arr[:, 0]
        mass = arr[:, 1]
        return mass.dot(intensity) / intensity.sum()

    def _as_array_query(self, session):
        anode = alias(ChromatogramTreeNode.__table__)
        bnode = alias(ChromatogramTreeNode.__table__)
        apeak = alias(DeconvolutedPeak.__table__)

        peak_join = apeak.join(
            ChromatogramTreeNodeToDeconvolutedPeak,
            ChromatogramTreeNodeToDeconvolutedPeak.c.peak_id == apeak.c.id)

        root_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == anode.c.id)

        branch_peaks_join = peak_join.join(
            anode,
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == anode.c.id).join(
            ChromatogramTreeNodeBranch,
            ChromatogramTreeNodeBranch.child_id == anode.c.id).join(
            bnode, ChromatogramTreeNodeBranch.parent_id == bnode.c.id).join(
            ChromatogramToChromatogramTreeNode,
            ChromatogramToChromatogramTreeNode.c.node_id == bnode.c.id)

        branch_intensities = select([apeak.c.intensity, anode.c.retention_time]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        root_intensities = select([apeak.c.intensity, anode.c.retention_time]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        all_intensities_q = root_intensities.union_all(branch_intensities).order_by(
            anode.c.retention_time)

        all_intensities = session.execute(all_intensities_q).fetchall()

        time = []
        signal = []
        current_signal = all_intensities[0][0]
        current_time = all_intensities[0][1]

        for intensity, rt in all_intensities[1:]:
            if abs(current_time - rt) < 1e-4:
                current_signal += intensity
            else:
                time.append(current_time)
                signal.append(current_signal)
                current_time = rt
                current_signal = intensity
        time.append(current_time)
        signal.append(current_signal)
        return np.array(time), np.array(signal)

    def as_arrays(self):
        session = object_session(self)
        return self._as_array_query(session)

    @property
    def weighted_neutral_mass(self):
        session = object_session(self)
        return self._weighted_neutral_mass_query(session)

    @property
    def total_signal(self):
        session = object_session(self)
        return self._as_array_query(session)[1].sum()

    @property
    def apex_time(self):
        time, intensity = self.as_arrays()
        return time[np.argmax(intensity)]

    def __repr__(self):
        return "DB" + repr(self.convert())

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def __iter__(self):
        return iter(self.nodes)


ChromatogramInterface.register(Chromatogram)


class ChromatogramWrapper(object):

    def _get_chromatogram(self):
        return self.chromatogram

    @property
    def adducts(self):
        return self._get_chromatogram().adducts

    @property
    def weighted_neutral_mass(self):
        return self._get_chromatogram().weighted_neutral_mass

    @property
    def total_signal(self):
        return self._get_chromatogram().total_signal

    def __len__(self):
        return len(self._get_chromatogram())

    def as_arrays(self):
        return self._get_chromatogram().as_arrays()

    @property
    def start_time(self):
        return self._get_chromatogram().start_time

    @property
    def end_time(self):
        return self._get_chromatogram().end_time

    @property
    def apex_time(self):
        return self._get_chromatogram().apex_time

    @property
    def neutral_mass(self):
        return self._get_chromatogram().neutral_mass

    @property
    def charge_states(self):
        return self._get_chromatogram().charge_states


ChromatogramInterface.register(ChromatogramWrapper)


ChromatogramToChromatogramTreeNode = Table(
    "ChromatogramToChromatogramTreeNode", Base.metadata,
    Column("chromatogram_id", Integer, ForeignKey(
        Chromatogram.id, ondelete="CASCADE"), primary_key=True),
    Column("node_id", Integer, ForeignKey(
        ChromatogramTreeNode.id, ondelete='CASCADE'), primary_key=True))


class CompositionGroup(Base, BoundToAnalysis):
    __tablename__ = "CompositionGroup"

    id = Column(Integer, primary_key=True)
    composition = Column(String(512), index=True)

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
        return "CompositionGroup(%r, %d)" % (self.convert(), self.id)


class CompositionGroupSerializer(SimpleSerializerCacheBase):
    _model_class = CompositionGroup


class ScoredChromatogram(object):
    score = Column(Numeric(8, 7, asdecimal=False), index=True)
    internal_score = Column(Numeric(8, 7, asdecimal=False))


class ChromatogramSolution(Base, BoundToAnalysis, ScoredChromatogram, ChromatogramWrapper):
    __tablename__ = "ChromatogramSolution"

    id = Column(Integer, primary_key=True)

    chromatogram_id = Column(Integer, ForeignKey(
        Chromatogram.id, ondelete='CASCADE'), index=True)
    composition_group_id = Column(Integer, ForeignKey(
        CompositionGroup.id, ondelete='CASCADE'), index=True)

    chromatogram = relationship(Chromatogram, lazy='joined')
    composition_group = relationship(CompositionGroup)

    def get_chromatogram(self):
        # return self.chromatogram.convert()
        return self.chromatogram.get_chromatogram()

    @property
    def key(self):
        if self.composition_group is not None:
            return self.composition_group._key()
        else:
            return self.neutral_mass

    # @property
    # def neutral_mass(self):
    #     return self.chromatogram.neutral_mass

    # @property
    # def start_time(self):
    #     return self.chromatogram.start_time

    # @property
    # def end_time(self):
    #     return self.chromatogram.end_time

    @property
    def composition(self):
        return self.composition_group

    def convert(self, *args, **kwargs):
        chromatogram_scoring_model = kwargs.pop(
            "chromatogram_scoring_model", GeneralScorer)
        chromatogram = self.chromatogram.convert(*args, **kwargs)
        composition = self.composition_group.convert() if self.composition_group else None
        chromatogram.composition = composition
        sol = MemoryChromatogramSolution(
            chromatogram, self.score, chromatogram_scoring_model, self.internal_score)
        used_as_adduct = []
        for pair in self.used_as_adduct:
            used_as_adduct.append((pair[0], pair[1].convert()))
        sol.used_as_adduct = used_as_adduct

        ambiguous_with = []
        for pair in self.ambiguous_with:
            ambiguous_with.append((pair[0], pair[1].convert()))
        sol.ambiguous_with = ambiguous_with
        return sol

    @property
    def used_as_adduct(self):
        pairs = []
        for rel in self._adducted_relationships_adduct:
            pairs.append((rel.owner.key, rel.mass_shift))
        return pairs

    @property
    def ambiguous_with(self):
        pairs = []
        for rel in self._adduct_relationships_owned:
            pairs.append((rel.adducted.key, rel.mass_shift))
        return pairs

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, composition_cache=None, node_peak_map=None,
                  *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = MassShiftSerializer(session)
        if composition_cache is None:
            composition_cache = CompositionGroupSerializer(session)
        db_composition_group = composition_cache.serialize(obj.composition)
        composition_group_id = db_composition_group.id if db_composition_group is not None else None
        inst = cls(
            composition_group_id=composition_group_id,
            analysis_id=analysis_id, score=obj.score,
            internal_score=obj.internal_score)
        chromatogram = Chromatogram.serialize(
            obj.chromatogram, session=session, analysis_id=analysis_id,
            peak_lookup_table=peak_lookup_table, mass_shift_cache=mass_shift_cache,
            scan_lookup_table=scan_lookup_table, node_peak_map=node_peak_map,
            *args, **kwargs)
        inst.chromatogram_id = chromatogram.id
        session.add(inst)
        session.flush()
        return inst

    # @property
    # def total_signal(self):
    #     return self.chromatogram.total_signal

    # @property
    # def weighted_neutral_mass(self):
    #     return self.chromatogram.weighted_neutral_mass

    def __repr__(self):
        return "DB" + repr(self.convert())

    # def __len__(self):
    #     return len(self.chromatogram)


class ChromatogramSolutionWrapper(ChromatogramWrapper):
    def _get_chromatogram(self):
        return self.solution

    @property
    def used_as_adduct(self):
        return self._get_chromatogram().used_as_adduct

    @property
    def ambiguous_with(self):
        return self._get_chromatogram().ambiguous_with


class GlycanCompositionChromatogram(Base, BoundToAnalysis, ScoredChromatogram, ChromatogramSolutionWrapper):
    __tablename__ = "GlycanCompositionChromatogram"

    id = Column(Integer, primary_key=True)

    chromatogram_solution_id = Column(
        Integer, ForeignKey(ChromatogramSolution.id, ondelete='CASCADE'), index=True)

    solution = relationship(ChromatogramSolution)

    glycan_composition_id = Column(
        Integer, ForeignKey(GlycanComposition.id, ondelete='CASCADE'), index=True)

    entity = relationship(GlycanComposition)

    def convert(self, *args, **kwargs):
        entity = self.entity.convert()
        solution = self.solution.convert(*args, **kwargs)
        case = solution.chromatogram.clone(MemoryGlycanCompositionChromatogram)
        case.composition = entity
        solution.chromatogram = case

        solution.id = self.id
        solution.solution_id = self.solution.id
        return solution

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, composition_cache=None, node_peak_map=None, *args, **kwargs):
        solution = ChromatogramSolution.serialize(
            obj, session, analysis_id, peak_lookup_table=peak_lookup_table,
            mass_shift_cache=mass_shift_cache, scan_lookup_table=scan_lookup_table,
            composition_cache=composition_cache, node_peak_map=node_peak_map,
            *args, **kwargs)
        inst = cls(
            chromatogram_solution_id=solution.id, glycan_composition_id=obj.composition.id,
            score=obj.score, internal_score=obj.internal_score, analysis_id=analysis_id)
        session.add(inst)
        session.flush()
        return inst

    @property
    def glycan_composition(self):
        return self.entity.convert()

    @property
    def composition(self):
        return self.glycan_composition

    def __repr__(self):
        return "DB" + repr(self.convert())


class UnidentifiedChromatogram(Base, BoundToAnalysis, ScoredChromatogram, ChromatogramSolutionWrapper):
    __tablename__ = "UnidentifiedChromatogram"

    id = Column(Integer, primary_key=True)

    chromatogram_solution_id = Column(
        Integer, ForeignKey(ChromatogramSolution.id, ondelete='CASCADE'), index=True)

    solution = relationship(ChromatogramSolution)

    def convert(self, *args, **kwargs):
        solution = self.solution.convert(*args, **kwargs)
        solution.id = self.id
        solution.solution_id = self.solution.id
        return solution

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, composition_cache=None, node_peak_map=None, *args, **kwargs):
        solution = ChromatogramSolution.serialize(
            obj, session, analysis_id, peak_lookup_table,
            mass_shift_cache, scan_lookup_table, composition_cache,
            node_peak_map=node_peak_map, *args, **kwargs)
        inst = cls(
            chromatogram_solution_id=solution.id,
            score=obj.score, internal_score=obj.internal_score,
            analysis_id=analysis_id)
        session.add(inst)
        session.flush()
        return inst

    def __repr__(self):
        return "DBUnidentified" + repr(self.convert())

    @property
    def glycan_composition(self):
        return None

    @property
    def composition(self):
        return None


class ChromatogramSolutionAdductedToChromatogramSolution(Base):
    __tablename__ = "ChromatogramSolutionAdductedToChromatogramSolution"

    adducted_solution_id = Column(Integer, ForeignKey(
        ChromatogramSolution.id, ondelete="CASCADE"), primary_key=True)
    mass_shift_id = Column(Integer, ForeignKey(
        CompoundMassShift.id, ondelete='CASCADE'), primary_key=True)
    owning_solution_id = Column(Integer, ForeignKey(
        ChromatogramSolution.id, ondelete="CASCADE"), primary_key=True)

    owner = relationship(ChromatogramSolution, backref=backref(
        "_adduct_relationships_owned"),
        foreign_keys=[owning_solution_id])
    adducted = relationship(ChromatogramSolution, backref=backref(
        "_adducted_relationships_adduct"),
        foreign_keys=[adducted_solution_id])
    mass_shift = relationship(CompoundMassShift)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.adducted.key} -{self.mass_shift.name}-> {self.owner.key})"
        return template.format(self=self)
