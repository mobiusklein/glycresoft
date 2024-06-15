from array import array
from collections import Counter

import numpy as np

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    Table, select, join, alias, bindparam)
from sqlalchemy.orm import relationship, backref, object_session, deferred
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext import baked

from glycresoft.chromatogram_tree import (
    ChromatogramTreeNode as MemoryChromatogramTreeNode,
    ChromatogramTreeList,
    Chromatogram as MemoryChromatogram,
    MassShift as MemoryMassShift,
    CompoundMassShift as MemoryCompoundMassShift,
    GlycanCompositionChromatogram as MemoryGlycanCompositionChromatogram,
    ChromatogramInterface, SimpleEntityChromatogram)

from glycresoft.chromatogram_tree.chromatogram import MIN_POINTS_FOR_CHARGE_STATE
from glycresoft.chromatogram_tree.utils import ArithmeticMapping

from glycresoft.scoring import (
    ChromatogramSolution as MemoryChromatogramSolution)
from glycresoft.models import GeneralScorer

from .analysis import BoundToAnalysis
from .hypothesis import GlycanComposition

from .spectrum import (
    Base, DeconvolutedPeak, MSScan, Mass, make_memory_deconvoluted_peak)

from glypy.composition.base import formula
from glypy import Composition


bakery = baked.bakery()


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

    def extract_key(self, obj):
        return extract_key(obj)

    def serialize(self, obj, *args, **kwargs):
        if obj is None:
            return None
        try:
            db_obj = self.store[self.extract_key(obj)]
            return db_obj
        except KeyError:
            db_obj = self._model_class.serialize(obj, self.session, *args, **kwargs)
            self.store[self.extract_key(db_obj)] = db_obj
            return db_obj

    def __getitem__(self, obj):
        return self.serialize(obj)


class MassShift(Base):
    __tablename__ = "MassShift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), index=True, unique=True)
    composition = Column(String(128))
    tandem_composition = deferred(Column(String(128)))

    _hash = None
    _mass = None
    _tandem_mass = None

    @property
    def mass(self):
        if self._mass is None:
            self._mass = Composition(str(self.composition)).mass
        return self._mass

    @property
    def tandem_mass(self):
        if self._tandem_mass is None:
            if self.tandem_composition:
                self._tandem_mass = Composition(str(self.tandem_composition)).mass
            else:
                self._tandem_mass = 0.0
        return self._tandem_mass

    def convert(self):
        try:
            tandem_composition = Composition(str(self.tandem_composition))
        except Exception:
            tandem_composition = None
        return MemoryMassShift(
            str(self.name), Composition(str(self.composition)), tandem_composition)

    @classmethod
    def serialize(cls, obj, session, *args, **kwargs):
        shift = session.query(MassShift).filter(MassShift.name == obj.name).all()
        if shift:
            return shift[0]
        else:
            db_obj = MassShift(
                name=obj.name, composition=formula(obj.composition),
                tandem_composition=formula(obj.tandem_composition))
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
        if self._hash is None:
            self._hash = hash(self.name)
        return self._hash


class CompoundMassShift(Base):
    __tablename__ = "CompoundMassShift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), index=True, unique=True)

    counts = association_proxy(
        "_counts", "count", creator=lambda k, v: MassShiftToCompoundMassShift(individual_id=k.id, count=v))

    _mass = None
    _tandem_mass = None

    @property
    def mass(self):
        if self._mass is None:
            self._mass = sum([k.mass * v for k, v in self.counts.items()])
        return self._mass

    @property
    def tandem_mass(self):
        if self._tandem_mass is None:
            self._tandem_mass = sum([k.tandem_mass * v for k, v in self.counts.items()])
        return self._tandem_mass

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

    def extract_key(self, obj):
        return obj.name


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

    def _get_child_nodes(self, session):
        query = bakery(lambda session: session.query(ChromatogramTreeNode).join(
            ChromatogramTreeNodeBranch.child).filter(
            ChromatogramTreeNodeBranch.parent_id == bindparam("parent_id")))
        result = query(session).params(parent_id=self.id).all()
        return result

    def _get_peaks(self, session):
        query = bakery(lambda session: session.query(DeconvolutedPeak).join(
            ChromatogramTreeNodeToDeconvolutedPeak).filter(
            ChromatogramTreeNodeToDeconvolutedPeak.c.node_id == bindparam("node_id")))
        result = query(session).params(node_id=self.id).all()
        return result

    def convert(self, node_type_cache=None, scan_id_cache=None):
        if scan_id_cache is not None:
            try:
                scan_id = scan_id_cache[self.scan_id]
            except KeyError:
                scan_id = self.scan.scan_id
                scan_id_cache[self.scan_id] = scan_id
        else:
            scan_id = self.scan.scan_id
        if node_type_cache is not None:
            try:
                node_type = node_type_cache[self.node_type_id]
            except KeyError:
                node_type = self.node_type.convert()
                node_type_cache[self.node_type_id] = node_type
        else:
            node_type = self.node_type.convert()

        session = object_session(self)
        # children = self.children
        children = self._get_child_nodes(session)
        # peaks = self.members
        peaks = self._get_peaks(session)

        inst = MemoryChromatogramTreeNode(
            self.retention_time, scan_id, [
                child.convert(node_type_cache, scan_id_cache) for child in children],
            [p.convert() for p in peaks],
            node_type)
        return inst

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, node_peak_map=None, *args, **kwargs):
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
                session.execute(ChromatogramTreeNodeToDeconvolutedPeak.insert(), [  # pylint: disable=no-value-for-parameter
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

    _total_signal_cache = None

    id = Column(Integer, primary_key=True, autoincrement=True)
    neutral_mass = Mass()

    @property
    def composition(self):
        return None

    start_time = Column(Numeric(8, 4, asdecimal=False), index=True)
    end_time = Column(Numeric(8, 4, asdecimal=False), index=True)

    nodes = relationship(ChromatogramTreeNode, secondary=lambda: ChromatogramToChromatogramTreeNode)

    @property
    def mass_shifts(self):
        session = object_session(self)
        return self._mass_shifts_query(session)

    def _mass_shifts_query(self, session):
        # (root_peaks_join, branch_peaks_join,
        #  anode, bnode, apeak) = _mass_shifts_query_inner

        # branch_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
        #     ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        # ).select_from(branch_peaks_join)

        # root_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
        #     ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        # ).select_from(root_peaks_join)

        # all_node_info_q = root_node_info.union_all(branch_node_info).order_by(
        #     anode.c.retention_time)

        # all_node_info = session.execute(all_node_info_q).fetchall()

        all_node_info = session.execute(_mass_shift_query_stmt, dict(id=self.id)).fetchall()

        node_type_ids = set()
        for node_type_id, rt in all_node_info:
            node_type_ids.add(node_type_id)

        node_types = []
        for ntid in node_type_ids:
            node_types.append(session.query(CompoundMassShift).get(ntid).convert())
        return node_types

    def mass_shift_signal_fractions(self):
        session = object_session(self)
        return self._mass_shift_signal_fraction_query(session)

    def bisect_mass_shift(self, mass_shift):
        session = object_session(self)

        (root_peaks_join, branch_peaks_join,
         anode, bnode, apeak) = _mass_shifts_query_inner
        root_node_info = select(
            [anode.c.node_type_id, anode.c.retention_time, apeak.c.intensity]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        branch_node_info = select(
            [anode.c.node_type_id, anode.c.retention_time, apeak.c.intensity]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        all_node_info_q = root_node_info.union_all(branch_node_info).order_by(anode.c.retention_time)
        mass_shift_id = session.query(CompoundMassShift.id).filter(CompoundMassShift.name == mass_shift.name).scalar()

        new_mass_shift = SimpleEntityChromatogram()
        new_no_mass_shift = SimpleEntityChromatogram()

        for r in sorted(session.execute(all_node_info_q), key=lambda x: x.retention_time):
            if r.node_type_id == mass_shift_id:
                new_mass_shift.setdefault(r.retention_time, 0)
                new_mass_shift[r.retention_time] += r.intensity
            else:
                new_no_mass_shift.setdefault(r.retention_time, 0)
                new_no_mass_shift[r.retention_time] += r.intensity
        new_mass_shift.mass_shifts = {mass_shift}
        new_no_mass_shift.mass_shifts = set(self.mass_shifts) - {mass_shift}

        return new_mass_shift, new_no_mass_shift

    def _mass_shift_signal_fraction_query(self, session):
        (root_peaks_join, branch_peaks_join,
         anode, bnode, apeak) = _mass_shifts_query_inner
        root_node_info = select([anode.c.node_type_id, anode.c.retention_time, apeak.c.intensity]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        branch_node_info = select([anode.c.node_type_id, anode.c.retention_time, apeak.c.intensity]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        all_node_info_q = root_node_info.union_all(branch_node_info).order_by(anode.c.retention_time)
        acc = ArithmeticMapping()
        for r in session.execute(all_node_info_q):
            acc[r.node_type_id] += r.intensity

        acc = ArithmeticMapping(
            {(session.query(CompoundMassShift).get(ntid).convert()): signal
             for ntid, signal in acc.items()})
        return acc

    def get_chromatogram(self):
        return self

    def _peaks_query(self, session):
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

        branch_ids = select([apeak.c.id]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(branch_peaks_join)

        root_ids = select([apeak.c.id]).where(
            ChromatogramToChromatogramTreeNode.c.chromatogram_id == self.id
        ).select_from(root_peaks_join)

        all_ids = root_ids.union_all(branch_ids)
        peaks = session.execute(all_ids).fetchall()
        return {p[0] for p in peaks}

    _peak_hash_ = None

    def _build_peak_hash(self):
        if self._peak_hash_ is None:
            session = object_session(self)
            self._peak_hash_ = frozenset(self._peaks_query(session))
        return self._peak_hash_

    @property
    def _peak_hash(self):
        return self._build_peak_hash()

    def is_distinct(self, other):
        return self._peak_hash.isdisjoint(other._peak_hash)

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

    def convert(self, node_type_cache=None, scan_id_cache=None, **kwargs):
        return self.orm_convert(
            node_type_cache=node_type_cache,
            scan_id_cache=scan_id_cache, **kwargs)

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
        session.execute(ChromatogramToChromatogramTreeNode.insert(), [  # pylint: disable=no-value-for-parameter
            {"chromatogram_id": inst.id, "node_id": node_id} for node_id in node_ids])
        return inst

    def _weighted_neutral_mass_query(self, session):
        all_intensity_mass = session.execute(_weighted_neutral_mass_stmt, dict(id=self.id)).fetchall()
        shift_ids = {}
        acc_mass = 0.0
        acc_intensity = 0.0
        shift_ids = {}
        for intensity, mass, shift_id in all_intensity_mass:
            try:
                delta_mass = shift_ids[shift_id]
            except KeyError:
                delta_mass = shift_ids[shift_id] = session.query(CompoundMassShift).get(shift_id).convert().mass
            acc_mass += (mass - delta_mass) * intensity
            acc_intensity += intensity
        return acc_mass / acc_intensity

    def _as_array_query(self, session):
        all_intensities = session.execute(_as_arrays_stmt, dict(id=self.id)).fetchall()

        time = array('d')
        signal = array('d')
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
        # Make sure intensity dtype doesn't undergo signed overflow by being
        # sufficiently wide to hold the value (e.g. not int32)
        return np.array(time), np.array(signal, dtype=np.float64)

    def as_arrays(self):
        session = object_session(self)
        return self._as_array_query(session)

    @property
    def weighted_neutral_mass(self):
        session = object_session(self)
        return self._weighted_neutral_mass_query(session)

    def integrate(self):
        time, intensity = self.as_arrays()
        return np.trapz(intensity, time)

    @property
    def total_signal(self):
        if self._total_signal_cache is None:
            session = object_session(self)
            self._total_signal_cache = self._as_array_query(session)[1].sum()
        return self._total_signal_cache

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


class MissingChromatogramError(ValueError):
    pass


class ChromatogramWrapper(object):

    def _get_chromatogram(self):
        if self.chromatogram is None:
            raise MissingChromatogramError()
        return self.chromatogram

    def has_chromatogram(self):
        return self.chromatogram is None

    def integrate(self):
        return self._get_chromatogram().integrate()

    @property
    def _peak_hash(self):
        try:
            return self._get_chromatogram()._peak_hash
        except MissingChromatogramError:
            return frozenset()

    def is_distinct(self, other):
        return self._peak_hash.isdisjoint(other._peak_hash)

    @property
    def mass_shifts(self):
        try:
            return self._get_chromatogram().mass_shifts
        except MissingChromatogramError:
            return []

    def mass_shift_signal_fractions(self):
        try:
            return self._get_chromatogram().mass_shift_signal_fractions()
        except MissingChromatogramError:
            return ArithmeticMapping()

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

    def bisect_mass_shift(self, mass_shift):
        return self._get_chromatogram().bisect_mass_shift(mass_shift)


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

    chromatogram = relationship(Chromatogram, lazy='select')
    composition_group = relationship(CompositionGroup)

    def get_chromatogram(self):
        return self.chromatogram.get_chromatogram()

    @property
    def key(self):
        if self.composition_group is not None:
            return self.composition_group._key()
        else:
            return self.neutral_mass

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
        used_as_mass_shift = []
        for pair in self.used_as_mass_shift:
            used_as_mass_shift.append((pair[0], pair[1].convert()))
        sol.used_as_mass_shift = used_as_mass_shift

        ambiguous_with = []
        for pair in self.ambiguous_with:
            ambiguous_with.append((pair[0], pair[1].convert()))
        sol.ambiguous_with = ambiguous_with
        return sol

    @property
    def used_as_mass_shift(self):
        pairs = []
        for rel in self._mass_shifted_relationships_mass_shift:
            pairs.append((rel.owner.key, rel.mass_shift))
        return pairs

    @property
    def ambiguous_with(self):
        pairs = []
        for rel in self._mass_shift_relationships_owned:
            pairs.append((rel.mass_shifted.key, rel.mass_shift))
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

    def __repr__(self):
        return "DB" + repr(self.convert())


class ChromatogramSolutionWrapper(ChromatogramWrapper):
    def _get_chromatogram(self):
        return self.solution

    @property
    def used_as_mass_shift(self):
        return self._get_chromatogram().used_as_mass_shift

    @property
    def ambiguous_with(self):
        return self._get_chromatogram().ambiguous_with

    def bisect_mass_shift(self, mass_shift):
        chromatogram = self._get_chromatogram()
        new_mass_shift, new_no_mass_shift = chromatogram.bisect_mass_shift(mass_shift)

        for chrom in (new_mass_shift, new_no_mass_shift):
            chrom.entity = self.entity
            chrom.composition = self.entity
            chrom.glycan_composition = self.glycan_composition
        return new_mass_shift, new_no_mass_shift


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

        try:
            solution.tandem_solutions = self.spectrum_cluster.convert()
        except AttributeError:
            solution.tandem_solutions = []

        return solution

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, composition_cache=None, node_peak_map=None,
                  *args, **kwargs):
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

    @property
    def key(self):
        return self.glycan_composition

    def __repr__(self):
        return "DB" + repr(self.convert())


class UnidentifiedChromatogram(Base, BoundToAnalysis, ScoredChromatogram, ChromatogramSolutionWrapper):
    __tablename__ = "UnidentifiedChromatogram"

    id = Column(Integer, primary_key=True)

    chromatogram_solution_id = Column(
        Integer, ForeignKey(ChromatogramSolution.id, ondelete='CASCADE'), index=True)

    solution = relationship(ChromatogramSolution)

    @property
    def key(self):
        return self.neutral_mass

    def convert(self, *args, **kwargs):
        solution = self.solution.convert(*args, **kwargs)
        solution.id = self.id
        solution.solution_id = self.solution.id

        try:
            solution.tandem_solutions = self.spectrum_cluster.convert()
        except AttributeError:
            solution.tandem_solutions = []

        return solution

    @classmethod
    def serialize(cls, obj, session, analysis_id, peak_lookup_table=None, mass_shift_cache=None,
                  scan_lookup_table=None, composition_cache=None, node_peak_map=None,
                  *args, **kwargs):
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


class ChromatogramSolutionMassShiftedToChromatogramSolution(Base):
    __tablename__ = "ChromatogramSolutionAdductToChromatogramSolution"

    mass_shifted_solution_id = Column(Integer, ForeignKey(
        ChromatogramSolution.id, ondelete="CASCADE"), primary_key=True)
    mass_shift_id = Column(Integer, ForeignKey(
        CompoundMassShift.id, ondelete='CASCADE'), primary_key=True)
    owning_solution_id = Column(Integer, ForeignKey(
        ChromatogramSolution.id, ondelete="CASCADE"), primary_key=True)

    owner = relationship(ChromatogramSolution, backref=backref(
        "_mass_shift_relationships_owned"),
        foreign_keys=[owning_solution_id])
    mass_shifted = relationship(ChromatogramSolution, backref=backref(
        "_mass_shifted_relationships_mass_shift"),
        foreign_keys=[mass_shifted_solution_id])
    mass_shift = relationship(CompoundMassShift)

    def __repr__(self):
        template = "{self.__class__.__name__}({self.mass_shifted.key} -{self.mass_shift.name}-> {self.owner.key})"
        return template.format(self=self)


# SQL Operation pre-definition. The Chromatogram table involves several complex joins to load
# relevant tree-like structure information. To reduce the amount of time SQLAlchemy spends
# on compiling these queries into strings, we use ``bindparam``-instrumented statements
# and expressions. These statements are made global.


def __build_mass_shifts_query_inner():

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
    return root_peaks_join, branch_peaks_join, anode, bnode, apeak


_mass_shifts_query_inner = __build_mass_shifts_query_inner()


def __build_mass_shifts_query():
    (root_peaks_join, branch_peaks_join,
        anode, bnode, apeak) = _mass_shifts_query_inner

    branch_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam('id')
    ).select_from(branch_peaks_join)

    root_node_info = select([anode.c.node_type_id, anode.c.retention_time]).where(
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam('id')
    ).select_from(root_peaks_join)

    all_node_info_q = root_node_info.union_all(branch_node_info).order_by(
        anode.c.retention_time)
    return all_node_info_q


_mass_shift_query_stmt = __build_mass_shifts_query()


def __build_weighted_neutral_mass_query():
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

    branch_intensities = select([apeak.c.intensity, apeak.c.neutral_mass, anode.c.node_type_id]).where(
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam("id")
    ).select_from(branch_peaks_join)

    root_intensities = select([apeak.c.intensity, apeak.c.neutral_mass, anode.c.node_type_id]).where(
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam("id")
    ).select_from(root_peaks_join)

    all_intensity_mass_q = root_intensities.union_all(branch_intensities)
    return all_intensity_mass_q


_weighted_neutral_mass_stmt = __build_weighted_neutral_mass_query()


def __build_as_arrays_query():
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
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam("id")
    ).select_from(branch_peaks_join)

    root_intensities = select([apeak.c.intensity, anode.c.retention_time]).where(
        ChromatogramToChromatogramTreeNode.c.chromatogram_id == bindparam("id")
    ).select_from(root_peaks_join)

    all_intensities_q = root_intensities.union_all(branch_intensities).order_by(
        anode.c.retention_time)

    return all_intensities_q


_as_arrays_stmt = __build_as_arrays_query()
