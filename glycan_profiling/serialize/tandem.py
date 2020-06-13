import warnings
from weakref import WeakValueDictionary

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table)
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import (
    relationship, backref, object_session, deferred, synonym, Load)
from sqlalchemy.exc import OperationalError

from glycan_profiling.tandem.spectrum_match import (
    SpectrumMatch as MemorySpectrumMatch,
    SpectrumSolutionSet as MemorySpectrumSolutionSet,
    ScoreSet, FDRSet, MultiScoreSpectrumMatch, MultiScoreSpectrumSolutionSet)

from glycan_profiling.structure import ScanInformation

from .analysis import BoundToAnalysis
from .hypothesis import Glycopeptide, GlycanComposition
from .chromatogram import CompoundMassShift
from .spectrum import (
    Base, MSScan)


def convert_scan_to_record(scan):
    return ScanInformation(
        scan.scan_id, scan.index, scan.scan_time,
        scan.ms_level, scan.precursor_information.convert())


class SpectrumMatchBase(BoundToAnalysis):

    score = Column(Numeric(12, 6, asdecimal=False), index=True)

    @declared_attr
    def scan_id(self):
        return Column(Integer, ForeignKey(MSScan.id), index=True)

    @declared_attr
    def scan(self):
        return relationship(MSScan)

    @declared_attr
    def mass_shift_id(self):
        return deferred(Column(Integer, ForeignKey(CompoundMassShift.id), index=True))

    @declared_attr
    def mass_shift(self):
        return relationship(CompoundMassShift, uselist=False)

    def _check_mass_shift(self, session):
        flag = session.info.get("has_spectrum_match_mass_shift")
        if flag:
            return self.mass_shift_id
        elif flag is None:
            try:
                session.begin_nested()
                shift_id = self.mass_shift_id
                session.rollback()
                session.info["has_spectrum_match_mass_shift"] = True
                return shift_id
            except OperationalError as err:
                session.rollback()
                warnings.warn(
                    ("Encountered an error while checking if "
                     "SpectrumMatch-tables support mass shifts: %r") % err)
                session.info["has_spectrum_match_mass_shift"] = False
                return None
        else:
            return None

    def _resolve_mass_shift(self, session, mass_shift_cache=None):
        mass_shift_id = self._check_mass_shift(session)
        if mass_shift_id is not None:
            if mass_shift_cache is not None:
                try:
                    mass_shift = mass_shift_cache[mass_shift_id]
                except KeyError:
                    mass_shift = self.mass_shift
                    mass_shift_cache[mass_shift_id] = mass_shift.convert()
                    mass_shift = mass_shift_cache[mass_shift_id]
                return mass_shift
            else:
                return self.mass_shift.convert()
        else:
            return None

    def is_multiscore(self):
        """Check whether this match has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return False


class SpectrumClusterBase(object):
    def __getitem__(self, i):
        return self.spectrum_solutions[i]

    def __iter__(self):
        return iter(self.spectrum_solutions)

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None, **kwargs):
        if scan_cache is None:
            scan_cache = dict()
        if mass_shift_cache is None:
            mass_shift_cache = dict()
        matches = [x.convert(mass_shift_cache=mass_shift_cache, scan_cache=scan_cache,
                             structure_cache=structure_cache, **kwargs) for x in self.spectrum_solutions]
        return matches

    def is_multiscore(self):
        """Check whether this match collection has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return self[0].is_multiscore()


class SolutionSetBase(object):

    @declared_attr
    def scan_id(self):
        return Column(Integer, ForeignKey(MSScan.id), index=True)

    @declared_attr
    def scan(self):
        return relationship(MSScan)

    @property
    def scan_time(self):
        return self.scan.scan_time

    def best_solution(self):
        if not self.is_multiscore():
            return sorted(self.spectrum_matches, key=lambda x: x.score, reverse=True)[0]
        else:
            return sorted(self.spectrum_matches, key=lambda x: (x.q_value, -x.score))[0]

    @property
    def score(self):
        return self.best_solution().score

    def __getitem__(self, i):
        return self.spectrum_matches[i]

    def __iter__(self):
        return iter(self.spectrum_matches)

    _target_map = None

    def _make_target_map(self):
        self._target_map = {
            sol.target: sol for sol in self
        }

    def solution_for(self, target):
        if self._target_map is None:
            self._make_target_map()
        return self._target_map[target]

    def is_multiscore(self):
        """Check whether this match has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return self[0].is_multiscore()


class GlycopeptideSpectrumCluster(Base, SpectrumClusterBase, BoundToAnalysis):
    __tablename__ = "GlycopeptideSpectrumCluster"

    id = Column(Integer, primary_key=True)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, *args, **kwargs):
        inst = cls()
        session.add(inst)
        session.flush()
        cluster_id = inst.id
        for solution_set in obj.tandem_solutions:
            GlycopeptideSpectrumSolutionSet.serialize(
                solution_set, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                cluster_id, *args, **kwargs)
        return inst


class GlycopeptideSpectrumSolutionSet(Base, SolutionSetBase, BoundToAnalysis):
    __tablename__ = "GlycopeptideSpectrumSolutionSet"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(
        Integer,
        ForeignKey(GlycopeptideSpectrumCluster.id, ondelete="CASCADE"),
        index=True)

    cluster = relationship(GlycopeptideSpectrumCluster, backref=backref("spectrum_solutions", lazy='subquery'))

    is_decoy = Column(Boolean, index=True)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                  cluster_id, is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            cluster_id=cluster_id)
        session.add(inst)
        session.flush()
        for solution in obj:
            GlycopeptideSpectrumMatch.serialize(
                solution, session, scan_look_up_cache, mass_shift_cache,
                analysis_id, inst.id, is_decoy, *args, **kwargs)
        return inst

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None, peptide_relation_cache=None):
        if scan_cache is None:
            scan_cache = dict()
        session = object_session(self)
        flag = session.info.get("has_spectrum_match_mass_shift")
        if flag:
            spectrum_match_q = self.spectrum_matches.options(
                Load(GlycopeptideSpectrumMatch).undefer("mass_shift_id")).all()
        else:
            spectrum_match_q = self.spectrum_matches.all()
        matches = [x.convert(mass_shift_cache=mass_shift_cache, scan_cache=scan_cache,
                             structure_cache=structure_cache, peptide_relation_cache=peptide_relation_cache)
                             for x in spectrum_match_q]
        matches.sort(key=lambda x: x.score, reverse=True)
        solution_set_tp = MemorySpectrumSolutionSet
        if matches and matches[0].is_multiscore():
            solution_set_tp = MultiScoreSpectrumSolutionSet
        inst = solution_set_tp(
            convert_scan_to_record(self.scan),
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
    solution_set = relationship(
        GlycopeptideSpectrumSolutionSet, backref=backref("spectrum_matches", lazy='dynamic'))

    q_value = Column(Numeric(8, 7, asdecimal=False), index=True)
    is_decoy = Column(Boolean, index=True)
    is_best_match = Column(Boolean, index=True)

    structure_id = Column(
        Integer, ForeignKey(Glycopeptide.id, ondelete='CASCADE'),
        index=True)

    structure = relationship(Glycopeptide)

    @property
    def target(self):
        return self.structure

    @property
    def q_value_set(self):
        return self.score_set

    def is_multiscore(self):
        """Check whether this match has been produced by summarizing a multi-score
        match, rather than a single score match.

        Returns
        -------
        bool
        """
        return self.score_set is not None

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                  solution_set_id, is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            score=obj.score,
            q_value=obj.q_value,
            solution_set_id=solution_set_id,
            is_best_match=obj.best_match,
            structure_id=obj.target.id,
            mass_shift_id=mass_shift_cache[obj.mass_shift].id)
        session.add(inst)
        session.flush()
        if hasattr(obj, 'score_set'):
            assert inst.id is not None
            GlycopeptideSpectrumMatchScoreSet.serialize_from_spectrum_match(obj, session, inst.id)
        return inst

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None, peptide_relation_cache=None):
        session = object_session(self)
        key = self.scan_id
        if scan_cache is None:
            scan = session.query(MSScan).get(key).convert(False, False)
        else:
            try:
                scan = scan_cache[key]
            except KeyError:
                scan = scan_cache[key] = session.query(MSScan).get(key).convert(False, False)

        key = self.structure_id
        if structure_cache is None:
            target = session.query(Glycopeptide).get(key).convert()
        else:
            try:
                target = structure_cache[key]
            except KeyError:
                target = structure_cache[key] = session.query(Glycopeptide).get(
                    key).convert(peptide_relation_cache=peptide_relation_cache)

        mass_shift = self._resolve_mass_shift(session, mass_shift_cache)
        if self.score_set is None:
            inst = MemorySpectrumMatch(scan, target, self.score, self.is_best_match,
                                       mass_shift=mass_shift)
        else:
            score_set, fdr_set = self.score_set.convert()
            inst = MultiScoreSpectrumMatch(
                scan, target, score_set, self.is_best_match,
                mass_shift=mass_shift, q_value_set=fdr_set, match_type=0)
        inst.q_value = self.q_value
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


class GlycanCompositionSpectrumCluster(Base, SpectrumClusterBase, BoundToAnalysis):
    __tablename__ = "GlycanCompositionSpectrumCluster"

    id = Column(Integer, primary_key=True)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, *args, **kwargs):
        inst = cls()
        session.add(inst)
        session.flush()
        cluster_id = inst.id
        for solution_set in obj.tandem_solutions:
            GlycanCompositionSpectrumSolutionSet.serialize(
                solution_set, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                cluster_id, *args, **kwargs)
        return inst

    source = relationship(
        "GlycanCompositionChromatogram",
        secondary=lambda: GlycanCompositionChromatogramToGlycanCompositionSpectrumCluster,
        backref=backref("spectrum_cluster", uselist=False))


class GlycanCompositionSpectrumSolutionSet(Base, SolutionSetBase, BoundToAnalysis):
    __tablename__ = "GlycanCompositionSpectrumSolutionSet"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(
        Integer,
        ForeignKey(GlycanCompositionSpectrumCluster.id, ondelete="CASCADE"),
        index=True)

    cluster = relationship(GlycanCompositionSpectrumCluster, backref=backref(
        "spectrum_solutions", lazy='subquery'))

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, cluster_id, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            analysis_id=analysis_id,
            cluster_id=cluster_id)
        session.add(inst)
        session.flush()
        # if we have a real SpectrumSolutionSet, then it will be iterable
        try:
            list(obj)
        except TypeError:
            # otherwise we have a single SpectrumMatch
            obj = [obj]
        for solution in obj:
            GlycanCompositionSpectrumMatch.serialize(
                solution, session, scan_look_up_cache, mass_shift_cache,
                analysis_id, inst.id, *args, **kwargs)
        return inst

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None):
        if scan_cache is None:
            scan_cache = dict()
        matches = [x.convert(mass_shift_cache=mass_shift_cache, scan_cache=scan_cache,
                             structure_cache=structure_cache) for x in self.spectrum_matches]
        matches.sort(key=lambda x: x.score, reverse=True)
        inst = MemorySpectrumSolutionSet(
            convert_scan_to_record(self.scan),
            matches
        )
        inst.q_value = min(x.q_value for x in inst)
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


class GlycanCompositionSpectrumMatch(Base, SpectrumMatchBase):
    __tablename__ = "GlycanCompositionSpectrumMatch"

    id = Column(Integer, primary_key=True)
    solution_set_id = Column(
        Integer, ForeignKey(
            GlycanCompositionSpectrumSolutionSet.id, ondelete='CASCADE'),
        index=True)
    solution_set = relationship(GlycanCompositionSpectrumSolutionSet,
                                backref=backref("spectrum_matches", lazy='subquery'))

    composition_id = Column(
        Integer, ForeignKey(GlycanComposition.id, ondelete='CASCADE'),
        index=True)

    composition = relationship(GlycanComposition)

    @property
    def target(self):
        return self.composition

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                  solution_set_id, is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            analysis_id=analysis_id,
            score=obj.score,
            solution_set_id=solution_set_id,
            composition_id=obj.target.id,
            mass_shift_id=mass_shift_cache[obj.mass_shift].id)
        session.add(inst)
        session.flush()
        return inst

    def convert(self, mass_shift_cache=None):
        session = object_session(self)
        scan = session.query(MSScan).get(self.scan_id).convert(False, False)
        mass_shift = self._resolve_mass_shift(session, mass_shift_cache)
        target = session.query(GlycanComposition).get(self.composition_id).convert()
        inst = MemorySpectrumMatch(scan, target, self.score, mass_shift=mass_shift)
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


class UnidentifiedSpectrumCluster(Base, SpectrumClusterBase, BoundToAnalysis):
    __tablename__ = "UnidentifiedSpectrumCluster"

    id = Column(Integer, primary_key=True)

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, *args, **kwargs):
        inst = cls()
        session.add(inst)
        session.flush()
        cluster_id = inst.id
        for solution_set in obj.tandem_solutions:
            UnidentifiedSpectrumSolutionSet.serialize(
                solution_set, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                cluster_id, *args, **kwargs)
        return inst

    source = relationship(
        "UnidentifiedChromatogram",
        secondary=lambda: UnidentifiedChromatogramToUnidentifiedSpectrumCluster,
        backref=backref("spectrum_cluster", uselist=False))


class UnidentifiedSpectrumSolutionSet(Base, SolutionSetBase, BoundToAnalysis):
    __tablename__ = "UnidentifiedSpectrumSolutionSet"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(
        Integer,
        ForeignKey(UnidentifiedSpectrumCluster.id, ondelete="CASCADE"),
        index=True)

    cluster = relationship(UnidentifiedSpectrumCluster, backref=backref(
        "spectrum_solutions", lazy='subquery'))

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, cluster_id, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            analysis_id=analysis_id,
            cluster_id=cluster_id)
        session.add(inst)
        session.flush()
        # if we have a real SpectrumSolutionSet, then it will be iterable
        try:
            list(obj)
        except TypeError:
            # otherwise we have a single SpectrumMatch
            obj = [obj]
        for solution in obj:
            UnidentifiedSpectrumMatch.serialize(
                solution, session, scan_look_up_cache, mass_shift_cache,
                analysis_id, inst.id, *args, **kwargs)
        return inst

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None):
        if scan_cache is None:
            scan_cache = dict()
        if mass_shift_cache is None:
            mass_shift_cache = dict()
        matches = [x.convert(mass_shift_cache=mass_shift_cache,
                             scan_cache=scan_cache) for x in self.spectrum_matches]
        matches.sort(key=lambda x: x.score, reverse=True)
        inst = MemorySpectrumSolutionSet(
            convert_scan_to_record(self.scan),
            matches
        )
        inst.q_value = min(x.q_value for x in inst)
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


class UnidentifiedSpectrumMatch(Base, SpectrumMatchBase):
    __tablename__ = "UnidentifiedSpectrumMatch"

    id = Column(Integer, primary_key=True)
    solution_set_id = Column(
        Integer, ForeignKey(
            UnidentifiedSpectrumSolutionSet.id, ondelete='CASCADE'),
        index=True)

    solution_set = relationship(UnidentifiedSpectrumSolutionSet,
                                backref=backref("spectrum_matches", lazy='subquery'))

    @classmethod
    def serialize(cls, obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, solution_set_id,
                  is_decoy=False, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            analysis_id=analysis_id,
            score=obj.score,
            solution_set_id=solution_set_id,
            mass_shift_id=mass_shift_cache[obj.mass_shift].id)
        session.add(inst)
        session.flush()
        return inst

    def convert(self, mass_shift_cache=None):
        session = object_session(self)
        scan = session.query(MSScan).get(self.scan_id).convert(False, False)
        mass_shift = self._resolve_mass_shift(session, mass_shift_cache)
        inst = MemorySpectrumMatch(scan, None, self.score, mass_shift=mass_shift)
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())


GlycanCompositionChromatogramToGlycanCompositionSpectrumCluster = Table(
    "GlycanCompositionChromatogramToGlycanCompositionSpectrumCluster", Base.metadata,
    Column("chromatogram_id", Integer, ForeignKey(
        "GlycanCompositionChromatogram.id", ondelete="CASCADE"), primary_key=True),
    Column("cluster_id", Integer, ForeignKey(
        GlycanCompositionSpectrumCluster.id, ondelete="CASCADE"), primary_key=True))


UnidentifiedChromatogramToUnidentifiedSpectrumCluster = Table(
    "UnidentifiedChromatogramToUnidentifiedSpectrumCluster", Base.metadata,
    Column("chromatogram_id", Integer, ForeignKey(
        "UnidentifiedChromatogram.id", ondelete="CASCADE"), primary_key=True),
    Column("cluster_id", Integer, ForeignKey(
        UnidentifiedSpectrumCluster.id, ondelete="CASCADE"), primary_key=True))


class GlycopeptideSpectrumMatchScoreSet(Base):
    __tablename__ = "GlycopeptideSpectrumMatchScoreSet"

    id = Column(Integer, ForeignKey(GlycopeptideSpectrumMatch.id), primary_key=True)
    peptide_score = Column(Numeric(12, 6, asdecimal=False), index=False)
    glycan_score = Column(Numeric(12, 6, asdecimal=False), index=False)
    glycopeptide_score = Column(Numeric(12, 6, asdecimal=False), index=False)
    glycan_coverage = deferred(Column(Numeric(8, 7, asdecimal=False), index=False))

    total_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    peptide_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    glycan_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    glycopeptide_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)

    spectrum_match = relationship(
        GlycopeptideSpectrumMatch, backref=backref("score_set", lazy='joined', uselist=False))

    @classmethod
    def serialize_from_spectrum_match(cls, obj, session, db_id):
        scores = obj.score_set
        qs = obj.q_value_set
        fields = dict(
            id=db_id,
            peptide_score=scores.peptide_score,
            glycan_score=scores.glycan_score,
            glycan_coverage=scores.glycan_coverage,
            glycopeptide_score=scores.glycopeptide_score,
            total_q_value=qs.total_q_value,
            peptide_q_value=qs.peptide_q_value,
            glycan_q_value=qs.glycan_q_value,
            glycopeptide_q_value=qs.glycopeptide_q_value
        )
        session.execute(cls.__table__.insert(), fields)

    def convert(self):
        score_set = ScoreSet(
            peptide_score=self.peptide_score,
            glycan_score=self.glycan_score,
            glycopeptide_score=self.glycopeptide_score,
            glycan_coverage=self.glycan_coverage)
        fdr_set = FDRSet(
            total_q_value=self.total_q_value,
            peptide_q_value=self.peptide_q_value,
            glycan_q_value=self.glycan_q_value,
            glycopeptide_q_value=self.glycopeptide_q_value)
        return score_set, fdr_set

    def __repr__(self):
        return "DB" + repr(self.convert())
