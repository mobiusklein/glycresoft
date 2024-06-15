from typing import Dict, Optional
from sqlalchemy import Column, Numeric, Integer, ForeignKey, select
from sqlalchemy.orm import relationship, backref, object_session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method

from glycresoft.serialize.analysis import BoundToAnalysis

from glycresoft.serialize.chromatogram import ChromatogramSolution, ChromatogramWrapper, CompoundMassShift, MissingChromatogramError

from glycresoft.serialize.tandem import (
    GlycopeptideSpectrumCluster,
    GlycopeptideSpectrumMatch,
    GlycopeptideSpectrumSolutionSet,
)

from glycresoft.serialize.hypothesis import Glycopeptide

from glycresoft.tandem.chromatogram_mapping import TandemAnnotatedChromatogram
from glycresoft.chromatogram_tree import GlycopeptideChromatogram
from glycresoft.chromatogram_tree.utils import ArithmeticMapping

from glycresoft.serialize.base import Base


class IdentifiedStructure(BoundToAnalysis, ChromatogramWrapper):
    @declared_attr
    def id(self):
        return Column(Integer, primary_key=True)

    @declared_attr
    def chromatogram_solution_id(self):
        return Column(Integer, ForeignKey(ChromatogramSolution.id, ondelete="CASCADE"), index=True)

    def has_chromatogram(self):
        return self.chromatogram_solution_id is not None

    @declared_attr
    def chromatogram(self):
        return relationship(ChromatogramSolution, lazy="select")

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

    structure_id = Column(Integer, ForeignKey(Glycopeptide.id, ondelete="CASCADE"), index=True)

    structure = relationship(Glycopeptide)

    ms1_score = Column(Numeric(8, 7, asdecimal=False), index=True)
    ms2_score = Column(Numeric(12, 6, asdecimal=False), index=True)
    q_value = Column(Numeric(8, 7, asdecimal=False), index=True)

    spectrum_cluster_id = Column(Integer, ForeignKey(GlycopeptideSpectrumCluster.id, ondelete="CASCADE"), index=True)

    spectrum_cluster = relationship(GlycopeptideSpectrumCluster, backref=backref("owners", lazy="dynamic"))

    ambiguous_id = Column(Integer, ForeignKey(AmbiguousGlycopeptideGroup.id, ondelete="CASCADE"), index=True)

    shared_with = relationship(AmbiguousGlycopeptideGroup, backref=backref("members", lazy="dynamic"))

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
    def localizations(self):
        return self.best_spectrum_match.localizations

    @property
    def best_spectrum_match(self) -> GlycopeptideSpectrumMatch:
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

    def is_multiscore(self) -> bool:
        """Check whether this match collection has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return self.spectrum_cluster.is_multiscore()

    @property
    def _best_spectrum_match(self):
        summary: IdentifiedGlycopeptideSummary = self._summary
        if summary is not None and summary.best_spectrum_match_id is not None:
            session = object_session(self)
            return session.query(GlycopeptideSpectrumMatch).get(summary.best_spectrum_match_id)

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
                        q_delta = abs(best_q_value - q_value)
                        best_q_value = q_value
                        if q_delta > 0.001:
                            best_score = match.score
                            best_match = match
                        else:
                            score = match.score
                            if score > best_score:
                                best_score = score
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
        expr = (
            select([Glycopeptide.is_multiply_glycosylated()])
            .where(Glycopeptide.id == IdentifiedGlycopeptide.structure_id)
            .label("is_multiply_glycosylated")
        )
        return expr

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
    def serialize(
        cls,
        obj,
        session,
        chromatogram_solution_id,
        tandem_cluster_id,
        analysis_id,
        build_summary: bool = False,
        *args,
        **kwargs,
    ):
        inst = cls(
            chromatogram_solution_id=chromatogram_solution_id,
            spectrum_cluster_id=tandem_cluster_id,
            analysis_id=analysis_id,
            q_value=obj.q_value,
            ms2_score=obj.ms2_score,
            ms1_score=obj.ms1_score,
            structure_id=obj.structure.id,
        )
        if build_summary:
            inst._build_summary()
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
    def bulk_convert(
        cls,
        iterable,
        expand_shared_with: bool = True,
        mass_shift_cache: Optional[Dict] = None,
        scan_cache: Optional[Dict] = None,
        structure_cache: Optional[Dict] = None,
        peptide_relation_cache: Optional[Dict] = None,
        shared_identification_cache: Optional[Dict] = None,
        min_q_value: float = 0.2,
        *args,
        **kwargs,
    ):
        session = object_session(iterable[0])
        if mass_shift_cache is None:
            mass_shift_cache = {m.id: m.convert() for m in session.query(CompoundMassShift).all()}
        if scan_cache is None:
            scan_cache = {}
        if structure_cache is None:
            structure_cache = {}
        if peptide_relation_cache is None:
            peptide_relation_cache = {}
        if shared_identification_cache is None:
            shared_identification_cache = {}
        result = [
            obj.convert(
                expand_shared_with=expand_shared_with,
                mass_shift_cache=mass_shift_cache,
                scan_cache=scan_cache,
                structure_cache=structure_cache,
                peptide_relation_cache=peptide_relation_cache,
                min_q_value=min_q_value,
                *args,
                **kwargs,
            )
            for obj in iterable
        ]
        return result

    def convert(
        self,
        expand_shared_with: bool = True,
        mass_shift_cache: Optional[Dict] = None,
        scan_cache: Optional[Dict] = None,
        structure_cache: Optional[Dict] = None,
        peptide_relation_cache: Optional[Dict] = None,
        shared_identification_cache: Optional[Dict] = None,
        min_q_value: float = 0.2,
        *args,
        **kwargs,
    ):
        # bind this name late to avoid circular import error
        from glycresoft.tandem.glycopeptide.identified_structure import (
            IdentifiedGlycopeptide as MemoryIdentifiedGlycopeptide,
        )

        session = object_session(self)
        if mass_shift_cache is None:
            mass_shift_cache = {m.id: m.convert() for m in session.query(CompoundMassShift).all()}
        if shared_identification_cache is None:
            shared_identification_cache = dict()
        if scan_cache is None:
            scan_cache = dict()
        if structure_cache is None:
            structure_cache = dict()
        if peptide_relation_cache is None:
            peptide_relation_cache = dict()
        if expand_shared_with and self.shared_with.members.count() > 1:
            stored = list(self.shared_with)
            converted = []
            for idgp in stored:
                if idgp.id in shared_identification_cache:
                    converted.append(shared_identification_cache[idgp.id])
                else:
                    tmp = idgp.convert(
                        expand_shared_with=False,
                        mass_shift_cache=mass_shift_cache,
                        scan_cache=scan_cache,
                        structure_cache=structure_cache,
                        peptide_relation_cache=peptide_relation_cache,
                        shared_identification_cache=shared_identification_cache,
                        min_q_value=min_q_value,
                        *args,
                        **kwargs,
                    )
                    shared_identification_cache[idgp.id] = tmp
                    converted.append(tmp)
            for i in range(len(stored)):
                converted[i].shared_with = converted[:i] + converted[i + 1 :]
            for i, member in enumerate(stored):
                if member.id == self.id:
                    return converted[i]
        else:
            spectrum_matches = GlycopeptideSpectrumSolutionSet.bulk_load_from_clusters(
                session,
                [self.spectrum_cluster_id],
                mass_shift_cache=mass_shift_cache,
                scan_cache=scan_cache,
                structure_cache=structure_cache,
                peptide_relation_cache=peptide_relation_cache,
                min_q_value=min_q_value,
            )[self.spectrum_cluster_id]
            if structure_cache is not None:
                if self.structure_id in structure_cache:
                    structure, _ = structure_cache[self.structure_id]
                else:
                    structure = Glycopeptide.bulk_load(
                        session,
                        [self.structure_id],
                        peptide_relation_cache=peptide_relation_cache,
                        structure_cache=structure_cache,
                    )[0]
            else:
                structure = self.structure.convert()

            chromatogram = self.chromatogram
            if chromatogram is not None:
                chromatogram = self.chromatogram.convert(*args, **kwargs)
                chromatogram.chromatogram = TandemAnnotatedChromatogram(
                    chromatogram.chromatogram.clone(GlycopeptideChromatogram)
                )
                chromatogram.chromatogram.tandem_solutions.extend(spectrum_matches)
                chromatogram.chromatogram.entity = structure

            inst = MemoryIdentifiedGlycopeptide(structure, spectrum_matches, chromatogram)
            inst.id = self.id
            shared_identification_cache[self.id] = inst
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

    def has_scan(self, scan_id: str) -> bool:
        return any([sset.scan.scan_id == scan_id for sset in self.tandem_solutions])

    def get_scan(self, scan_id: str):
        for sset in self.tandem_solutions:
            if sset.scan.scan_id == scan_id:
                return sset
        raise KeyError(scan_id)

    @declared_attr
    def _summary(self):
        return relationship("IdentifiedGlycopeptideSummary", lazy="joined", uselist=False)

    @property
    def total_signal(self):
        summary = self._summary
        if summary is not None:
            return summary.total_signal
        try:
            return self.chromatogram.total_signal
        except AttributeError:
            return 0

    @property
    def apex_time(self):
        summary = self._summary
        if summary is not None and summary.apex_time is not None:
            return summary.apex_time
        return super().apex_time

    @property
    def start_time(self):
        summary = self._summary
        if summary is not None and summary.start_time is not None:
            return summary.start_time
        return super().start_time

    @property
    def end_time(self):
        summary = self._summary
        if summary is not None and summary.end_time is not None:
            return summary.end_time
        return super().end_time

    @property
    def weighted_neutral_mass(self):
        summary = self._summary
        if summary is not None and summary.weighted_neutral_mass is not None:
            return summary.weighted_neutral_mass
        return super().weighted_neutral_mass

    def _build_summary(self):
        if self._summary is not None:
            return
        session = object_session(self)

        try:
            apex_time = self.apex_time
            total_signal = self.total_signal
            start_time = self.start_time
            end_time = self.end_time
        except MissingChromatogramError:
            apex_time = total_signal = start_time = end_time = None
        best_spectrum_match_id = self.best_spectrum_match.id

        summary = IdentifiedGlycopeptideSummary(
            id=self.id,
            weighted_neutral_mass=self.weighted_neutral_mass,
            apex_time=apex_time,
            total_signal=total_signal,
            start_time=start_time,
            end_time=end_time,
            best_spectrum_match_id=best_spectrum_match_id,
        )
        session.add(summary)
        return summary


class IdentifiedGlycopeptideSummary(Base):
    __tablename__ = "IdentifiedGlycopeptideSummary"

    id = Column(Integer, ForeignKey(IdentifiedGlycopeptide.id), primary_key=True)

    weighted_neutral_mass = Column(Numeric(12, 6, asdecimal=False), index=True)
    apex_time = Column(Numeric(12, 6, asdecimal=False), index=True)
    total_signal = Column(Numeric(12, 6, asdecimal=False), index=True)
    start_time = Column(Numeric(12, 6, asdecimal=False), index=True)
    end_time = Column(Numeric(12, 6, asdecimal=False), index=True)
    best_spectrum_match_id = Column(
        Integer, ForeignKey(GlycopeptideSpectrumMatch.id, ondelete="CASCADE"), index=True, nullable=True
    )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.id}, weighted_neutral_mass={self.weighted_neutral_mass}, apex_time={self.apex_time}, "
            f"total_signal={self.total_signal}, start_time={self.start_time}, end_time={self.end_time}, "
            f"best_spectrum_match_id={self.best_spectrum_match_id})"
        )
