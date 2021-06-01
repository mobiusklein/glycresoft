from sqlalchemy import (
    Column, Numeric, Integer, ForeignKey, select)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

from glycan_profiling.serialize.analysis import BoundToAnalysis

from glycan_profiling.serialize.chromatogram import (
    ChromatogramSolution,
    ChromatogramWrapper)

from glycan_profiling.serialize.tandem import (
    GlycopeptideSpectrumCluster)

from glycan_profiling.serialize.hypothesis import Glycopeptide

from glycan_profiling.tandem.chromatogram_mapping import TandemAnnotatedChromatogram
from glycan_profiling.chromatogram_tree import GlycopeptideChromatogram
from glycan_profiling.chromatogram_tree.utils import ArithmeticMapping

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

    def bisect_mass_shift(self, mass_shift):
        chromatogram = self._get_chromatogram()
        new_mass_shift, new_no_mass_shift = chromatogram.bisect_mass_shift(mass_shift)

        for chrom in (new_mass_shift, new_no_mass_shift):
            chrom.entity = self.entity
            chrom.composition = self.entity
            chrom.glycan_composition = self.glycan_composition
        return new_mass_shift, new_no_mass_shift


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
    def score_set(self):
        """The :class:`~.ScoreSet` of the best MS/MS match

        Returns
        -------
        :class:`~.ScoreSet`
        """
        if not self.is_multiscore():
            return None
        best_match = self._best_spectrum_match
        if best_match is None:
            return None
        return best_match.score_set

    @property
    def best_spectrum_match(self):
        return self._best_spectrum_match

    @property
    def q_value_set(self):
        """The :class:`~.FDRSet` of the best MS/MS match

        Returns
        -------
        :class:`~.FDRSet`
        """
        if not self.is_multiscore():
            return None
        best_match = self._best_spectrum_match
        if best_match is None:
            return None
        return best_match.q_value_set

    def is_multiscore(self):
        """Check whether this match collection has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return self.spectrum_cluster.is_multiscore()

    @property
    def _best_spectrum_match(self):
        is_multiscore = self.is_multiscore()
        best_match = None
        if is_multiscore:
            best_score = 0.0
            best_q_value = 1.0
        else:
            best_score = 0.0
        for solution_set in self.spectrum_cluster:
            try:
                match = solution_set.solution_for(self.structure)
                if is_multiscore:
                    q_value = match.q_value
                    if q_value <= best_q_value:
                        delta_q_value = best_q_value - q_value
                        best_q_value = q_value
                        score = match.score
                        if score > best_score:
                            best_score = score
                            best_match = match
                        elif delta_q_value > 0.001:
                            best_match = match
                else:
                    score = match.score
                    if score > best_score:
                        best_score = score
                        best_match = match
            except KeyError:
                continue
        return best_match

    @hybrid_method
    def is_multiply_glycosylated(self):
        return self.structure.is_multiply_glycosylated()

    @is_multiply_glycosylated.expression
    def is_multiply_glycosylated(self):
        expr = select([Glycopeptide.is_multiply_glycosylated()]).where(
            Glycopeptide.id == IdentifiedGlycopeptide.structure_id).label(
            "is_multiply_glycosylated")
        return

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
        return self.spectrum_cluster.spectrum_solutions

    def mass_shift_tandem_solutions(self):
        mapping = ArithmeticMapping()
        for sm in self.tandem_solutions:
            try:
                mapping[sm.solution_for(self.structure).mass_shift] += 1
            except KeyError:
                continue
        return mapping

    def get_chromatogram(self, *args, **kwargs):
        chromatogram = self.chromatogram.convert(*args, **kwargs)
        chromatogram.chromatogram = chromatogram.chromatogram.clone(GlycopeptideChromatogram)
        structure = self.structure.convert()
        chromatogram.chromatogram.entity = structure
        return chromatogram

    @classmethod
    def bulk_convert(cls, iterable, expand_shared_with=True, mass_shift_cache=None, scan_cache=None,
                     structure_cache=None, peptide_relation_cache=None, *args, **kwargs):
        if mass_shift_cache is None:
            mass_shift_cache = {}
        if scan_cache is None:
            scan_cache = {}
        if structure_cache is None:
            structure_cache = {}
        if peptide_relation_cache is None:
            peptide_relation_cache = {}
        result = [
            obj.convert(expand_shared_with=expand_shared_with,
                        mass_shift_cache=mass_shift_cache,
                        scan_cache=scan_cache,
                        structure_cache=structure_cache,
                        peptide_relation_cache=peptide_relation_cache,
                        *args, **kwargs)
            for obj in iterable]
        return result


    def convert(self, expand_shared_with=True, mass_shift_cache=None, scan_cache=None, structure_cache=None,
                peptide_relation_cache=None, *args, **kwargs):
        # bind this name late to avoid circular import error
        from glycan_profiling.tandem.glycopeptide.identified_structure import (
            IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide)

        if mass_shift_cache is None:
            mass_shift_cache = dict()
        if scan_cache is None:
            scan_cache = dict()
        if structure_cache is None:
            structure_cache = dict()
        if peptide_relation_cache is None:
            peptide_relation_cache = dict()
        if expand_shared_with and self.shared_with:
            stored = list(self.shared_with)
            converted = [
                x.convert(expand_shared_with=False, mass_shift_cache=mass_shift_cache,
                          scan_cache=scan_cache, structure_cache=structure_cache,
                          peptide_relation_cache=peptide_relation_cache, * args, **kwargs) for x in stored]
            for i in range(len(stored)):
                converted[i].shared_with = converted[:i] + converted[i + 1:]
            for i, member in enumerate(stored):
                if member.id == self.id:
                    return converted[i]
        else:
            spectrum_matches = self.spectrum_cluster.convert(
                mass_shift_cache=mass_shift_cache, scan_cache=scan_cache,
                structure_cache=structure_cache, peptide_relation_cache=peptide_relation_cache)
            if structure_cache is not None:
                try:
                    structure = structure_cache[self.structure_id]
                except KeyError:
                    structure = structure_cache[self.structure_id] = self.structure.convert()
            else:
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
