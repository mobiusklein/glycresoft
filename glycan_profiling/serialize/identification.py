from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr

from glycan_profiling.serialize.analysis import BoundToAnalysis

from glycan_profiling.serialize.chromatogram import (
    ChromatogramSolution,
    ChromatogramWrapper)

from glycan_profiling.serialize.tandem import (
    GlycopeptideSpectrumCluster)

from glycan_profiling.serialize.hypothesis import Glycopeptide

from glycan_profiling.tandem.glycopeptide.identified_structure import (
    IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide)

from glycan_profiling.tandem.chromatogram_mapping import TandemAnnotatedChromatogram
from glycan_profiling.chromatogram_tree import GlycopeptideChromatogram

from glycan_profiling.serialize.base import (
    Base)


class IdentifiedStructure(BoundToAnalysis, ChromatogramWrapper):
    @declared_attr
    def id(self):
        return Column(Integer, primary_key=True)

    @declared_attr
    def chromatogram_solution_id(self):
        return Column(Integer, ForeignKey(ChromatogramSolution.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def chromatogram(self):
        return relationship(ChromatogramSolution, lazy='joined')

    def get_chromatogram(self):
        return self.chromatogram.get_chromatogram()

    @property
    def total_signal(self):
        try:
            return self.chromatogram.total_signal
        except AttributeError:
            return 0

    @property
    def composition(self):
        raise NotImplementedError()

    @property
    def entity(self):
        raise NotImplementedError()


class AmbiguousGlycopeptideGroup(Base, BoundToAnalysis):
    __tablename__ = "AmbiguousGlycopeptideGroup"

    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return "AmbiguousGlycopeptideGroup(%d)" % (self.id,)

    @classmethod
    def serialize(cls, members, session, analysis_id, *args, **kwargs):
        inst = cls(analysis_id=analysis_id)
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
        Integer, ForeignKey(Glycopeptide.id, ondelete='CASCADE'),
        index=True)

    structure = relationship(Glycopeptide)

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

    @property
    def glycan_composition(self):
        return self.structure.glycan_composition

    @property
    def protein_relation(self):
        return self.structure.protein_relation

    @property
    def start_position(self):
        return self.protein_relation.start_position

    @property
    def end_position(self):
        return self.protein_relation.end_position

    def overlaps(self, other):
        return self.protein_relation.overlaps(other.protein_relation)

    def spans(self, position):
        return position in self.protein_relation

    @property
    def spectrum_matches(self):
        return self.spectrum_cluster.spectrum_solutions

    @classmethod
    def serialize(cls, obj, session, chromatogram_solution_id, tandem_cluster_id, analysis_id, *args, **kwargs):
        inst = cls(
            chromatogram_solution_id=chromatogram_solution_id,
            spectrum_cluster_id=tandem_cluster_id,
            analysis_id=analysis_id,
            q_value=obj.q_value,
            ms2_score=obj.ms2_score,
            ms1_score=obj.ms1_score,
            structure_id=obj.structure.id)
        session.add(inst)
        session.flush()
        return inst

    @property
    def tandem_solutions(self):
        return self.spectrum_cluster

    def get_chromatogram(self, *args, **kwargs):
        chromatogram = self.chromatogram.convert(*args, **kwargs)
        chromatogram.chromatogram = chromatogram.chromatogram.clone(GlycopeptideChromatogram)
        structure = self.structure.convert()
        chromatogram.chromatogram.entity = structure
        return chromatogram

    def convert(self, expand_shared_with=True, *args, **kwargs):
        if expand_shared_with and self.shared_with:
            stored = list(self.shared_with)
            converted = [x.convert(expand_shared_with=False, *args, **kwargs) for x in stored]
            for i in range(len(stored)):
                converted[i].shared_with = converted[:i] + converted[i + 1:]
            for i, member in enumerate(stored):
                if member.id == self.id:
                    return converted[i]
        else:
            spectrum_matches = self.spectrum_cluster.convert()
            structure = self.structure.convert()

            chromatogram = self.chromatogram
            if chromatogram is not None:
                chromatogram = self.chromatogram.convert(*args, **kwargs)
                chromatogram.chromatogram = TandemAnnotatedChromatogram(
                    chromatogram.chromatogram.clone(GlycopeptideChromatogram))
                chromatogram.chromatogram.tandem_solutions.extend(spectrum_matches)
                chromatogram.chromatogram.entity = structure

            inst = MemoryIdentifiedGlycopeptide(structure, spectrum_matches, chromatogram)
            inst.id = self.id
            return inst

    @property
    def composition(self):
        return self.structure.convert()

    @property
    def entity(self):
        return self.structure.convert()

    def __repr__(self):
        return "DB" + repr(self.convert())

    def __str__(self):
        return str(self.structure)
