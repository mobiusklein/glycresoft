from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.declarative import declared_attr

from .analysis import BoundToAnalysis
from .chromatogram import ChromatogramSolution
from .tandem import GlycopeptideSpectrumCluster

from glycan_profiling.tandem.glycopeptide.identified_structure import (
    IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide)

from glycan_profiling.tandem.spectrum_matcher_base import (
    TargetReference)

from ms_deisotope.output.db import (
    Base)


class IdentifiedStructure(BoundToAnalysis):
    @declared_attr
    def id(self):
        return Column(Integer, primary_key=True)

    @declared_attr
    def chromatogram_solution_id(self):
        return Column(Integer, ForeignKey(ChromatogramSolution.id), index=True)

    @declared_attr
    def chromatogram(self):
        return relationship(ChromatogramSolution)


class AmbiguousGlycopeptideGroup(Base):
    __tablename__ = "AmbiguousGlycopeptideGroup"

    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return "AmbiguousGlycopeptideGroup(%d)" % (self.id,)

    @classmethod
    def serialize(cls, members, session, *args, **kwargs):
        inst = cls()
        session.add(inst)
        session.flush()
        for member in members:
            member.ambiguous_id = inst.id
            session.add(member)
        session.flush()
        return inst

    def __iter__(self):
        return iter(self.members)


class IdentifiedGlycopeptide(Base, IdentifiedStructure):
    __tablename__ = "IdentifiedGlycopeptide"

    structure_id = Column(
        Integer,  # ForeignKey(, ondelte='CASCADE')
        index=True)

    ms1_score = Column(Numeric(8, 7, asdecimal=False), index=True)
    ms2_score = Column(Numeric(12, 6, asdecimal=False), index=True)
    q_value = Column(Numeric(8, 7, asdecimal=False), index=True)

    spectrum_cluster_id = Column(
        Integer,
        ForeignKey(GlycopeptideSpectrumCluster.id, ondelete='CASCADE'),
        index=True)

    spectrum_cluster = relationship(GlycopeptideSpectrumCluster)

    ambiguous_id = Column(Integer, ForeignKey(
        AmbiguousGlycopeptideGroup.id, ondelete='CASCADE'), index=True)

    shared_with = relationship(
        AmbiguousGlycopeptideGroup, backref=backref(
            "members", lazy='dynamic'))

    @classmethod
    def serialize(cls, obj, session, chromatogram_solution_id, tandem_cluster_id, *args, **kwargs):
        inst = cls(
            chromatogram_solution_id=chromatogram_solution_id,
            spectrum_cluster_id=tandem_cluster_id,
            q_value=obj.q_value,
            ms2_score=obj.ms2_score,
            ms1_score=obj.ms1_score,
            structure_id=obj.structure.id)
        session.add(inst)
        session.flush()
        return inst

    def convert(self, expand_shared_with=True):
        if expand_shared_with and self.shared_with:
            stored = list(self.shared_with)
            converted = [x.convert(expand_shared_with=False) for x in stored]
            for i in range(len(stored)):
                converted[i].shared_with = converted[:i] + converted[i + 1:]
            for i, member in enumerate(stored):
                if member.id == self.id:
                    return converted[i]
        else:
            chromatogram = self.chromatogram.convert()
            # FIXME: When structures are migrated into this database
            # schema, remove the reference shim
            structure = TargetReference(self.structure_id)
            structure.protein_relation = None
            spectrum_matches = self.spectrum_cluster.convert()
            inst = MemoryIdentifiedGlycopeptide(structure, spectrum_matches, chromatogram)
            return inst

    def __repr__(self):
        return "DB" + repr(self.convert())
