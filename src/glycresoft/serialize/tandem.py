import warnings

from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Deque

from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    Boolean, Table, select)
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import (
    relationship, backref, object_session, Load)
from sqlalchemy.exc import OperationalError

from glycresoft.tandem.spectrum_match import (
    SpectrumMatch as MemorySpectrumMatch,
    SpectrumSolutionSet as MemorySpectrumSolutionSet,
    ScoreSet, FDRSet, MultiScoreSpectrumMatch,
    MultiScoreSpectrumSolutionSet, LocalizationScore as MemoryLocalizationScore)

from glycresoft.structure import ScanInformation

from .analysis import BoundToAnalysis
from .hypothesis import Glycopeptide, GlycanComposition
from .chromatogram import CompoundMassShift
from .spectrum import (
    Base, MSScan)
from .utils import chunkiter, needs_migration

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
        return needs_migration(Column(Integer, ForeignKey(CompoundMassShift.id), index=True))

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

    def __len__(self):
        return len(self.spectrum_solutions)

    def __repr__(self):
        template = "{self.__class__.__name__}(<{size}>)"
        return template.format(self=self, size=len(self))

    @classmethod
    def compute_key(cls, members) -> frozenset:
        return frozenset([m.key for m in members])

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None, **kwargs):
        if scan_cache is None:
            scan_cache = dict()
        if mass_shift_cache is None:
            mass_shift_cache = dict()
        matches = [x.convert(mass_shift_cache=mass_shift_cache, scan_cache=scan_cache,
                             structure_cache=structure_cache, **kwargs) for x in self.spectrum_solutions]
        return matches

    def is_multiscore(self) -> bool:
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

    def best_solution(self) -> SpectrumMatchBase:
        best_match = self.spectrum_matches.filter_by(is_best_match=True).first()
        if best_match is not None:
            # If the best match is already marked, return it
            return best_match
        if not self.is_multiscore():
            return sorted(self.spectrum_matches, key=lambda x: x.score, reverse=True)[0]
        else:
            return sorted(self.spectrum_matches, key=lambda x: (x.score_set.convert()[0]), reverse=True)[0]

    def is_ambiguous(self) -> bool:
        seen = set()

        for sm in self.spectrum_matches.filter_by(is_best_match=True).all():
            seen.add(str(sm.target))
        return len(seen) > 1

    @property
    def key(self) -> frozenset:
        scan_id = self.scan.id
        return frozenset([(scan_id, match.target.id) for match in self])

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
    def serialize(cls, obj: MemorySpectrumSolutionSet, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                  cluster_id, is_decoy=False, *args, **kwargs):
        obj.rank()
        gpsm = obj.best_solution()
        if gpsm is None:
            gpsm = obj[0]
        if not gpsm.best_match:
            obj.mark_top_solutions()
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            cluster_id=cluster_id)
        session.add(inst)
        session.flush()
        GlycopeptideSpectrumMatch.serialize_bulk(
            obj, session, scan_look_up_cache, mass_shift_cache, analysis_id, inst.id)
        return inst

    def convert(self, mass_shift_cache=None, scan_cache=None, structure_cache=None, peptide_relation_cache=None):
        if scan_cache is None:
            scan_cache = {}
        if mass_shift_cache is None:
            mass_shift_cache = {}
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
        solution_set_tp = MemorySpectrumSolutionSet
        if matches and matches[0].is_multiscore():
            solution_set_tp = MultiScoreSpectrumSolutionSet
        inst = solution_set_tp(
            convert_scan_to_record(self.scan),
            matches
        )
        inst._is_sorted = True
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())

    @classmethod
    def bulk_load(cls, session,
                  ids,
                  chunk_size: int=512,
                  mass_shift_cache=None,
                  scan_cache=None,
                  structure_cache=None,
                  peptide_relation_cache=None,
                  flatten=True,
                  min_q_value=0.2):
        return cls._inner_bulk_load(
            cls.id.in_,
            session,
            ids,
            chunk_size,
            mass_shift_cache,
            scan_cache,
            structure_cache,
            peptide_relation_cache,
            flatten,
            min_q_value=min_q_value
        )

    @classmethod
    def bulk_load_from_clusters(cls, session,
                                ids,
                                chunk_size: int = 512,
                                mass_shift_cache=None,
                                scan_cache=None,
                                structure_cache=None,
                                peptide_relation_cache=None,
                                min_q_value=0.2
                            ):
        by_id = cls._inner_bulk_load(
            cls.cluster_id.in_,
            session,
            ids,
            chunk_size,
            mass_shift_cache,
            scan_cache,
            structure_cache,
            peptide_relation_cache,
            flatten=False,
            min_q_value=min_q_value
        )
        groups = DefaultDict(list)
        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(select([cls.id, cls.cluster_id]).where(cls.cluster_id.in_(chunk)))
            for (sset_id, cluster_id) in res.fetchall():
                groups[cluster_id].append(by_id[sset_id])
        return groups

    @classmethod
    def _inner_bulk_load(
        cls,
        selector,
        session,
        ids,
        chunk_size: int = 512,
        mass_shift_cache=None,
        scan_cache=None,
        structure_cache=None,
        peptide_relation_cache=None,
        flatten=True,
        min_q_value=0.2
    ):
        if selector is None:
            selector = cls.id.in_
        if scan_cache is None:
            scan_cache = {}
        if mass_shift_cache is None:
            mass_shift_cache = {
                m.id: m.convert() for m in session.query(CompoundMassShift).all()
            }
        out = {}
        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(cls.__table__.select().where(
                selector(chunk)))
            for self in res.fetchall():
                gpsm_ids = [
                    i[0] for i in session.query(GlycopeptideSpectrumMatch.id).filter(
                        GlycopeptideSpectrumMatch.solution_set_id == self.id,
                        GlycopeptideSpectrumMatch.q_value <= min_q_value
                    ).all()
                ]
                if not gpsm_ids:
                    gpsm_ids = [
                        i[0] for i in session.query(GlycopeptideSpectrumMatch.id).filter(
                            GlycopeptideSpectrumMatch.solution_set_id == self.id,
                        ).order_by(
                            GlycopeptideSpectrumMatch.q_value.asc(),
                            GlycopeptideSpectrumMatch.score.desc()
                        ).limit(5)
                    ]
                matches = GlycopeptideSpectrumMatch.bulk_load(
                    session,
                    gpsm_ids,
                    chunk_size,
                    mass_shift_cache=mass_shift_cache,
                    scan_cache=scan_cache,
                    structure_cache=structure_cache,
                    peptide_relation_cache=peptide_relation_cache
                )
                matches = [matches[k] for k in gpsm_ids]
                solution_set_tp = MemorySpectrumSolutionSet
                if matches and matches[0].is_multiscore():
                    solution_set_tp = MultiScoreSpectrumSolutionSet

                if self.scan_id not in scan_cache:
                    MSScan.bulk_load(session, [self.scan_id], scan_cache=scan_cache)
                inst = solution_set_tp(scan_cache[self.scan_id], matches)
                inst._is_sorted = True
                inst.id = self.id
                out[self.id] = inst
        if flatten:
            return [out[i] for i in ids]
        return out


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
    is_decoy = Column(Boolean)
    is_best_match = Column(Boolean, index=True)
    rank = needs_migration(Column(Integer, default=0), 0)

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
                  solution_set_id, is_decoy=False, save_score_set=True, *args, **kwargs):
        inst = cls(
            scan_id=scan_look_up_cache[obj.scan.id],
            is_decoy=is_decoy,
            analysis_id=analysis_id,
            score=obj.score,
            rank=obj.rank,
            q_value=obj.q_value,
            solution_set_id=solution_set_id,
            is_best_match=obj.best_match,
            structure_id=obj.target.id,
            mass_shift_id=mass_shift_cache[obj.mass_shift].id)
        session.add(inst)
        session.flush()
        if hasattr(obj, 'score_set') and save_score_set:
            assert inst.id is not None
            GlycopeptideSpectrumMatchScoreSet.serialize_from_spectrum_match(obj, session, inst.id)
            if obj.localizations:
                locs = (LocalizationScore.get_fields_from_object(obj, inst.id))
                if locs:
                    session.bulk_insert_mappings(LocalizationScore, locs)

        return inst

    @classmethod
    def serialize_bulk(cls, objs, session, scan_look_up_cache, mass_shift_cache, analysis_id,
                       solution_set_id, is_decoy=False, save_score_set=True, *args, **kwargs):
        acc = []
        revmap: Dict[Any, MemorySpectrumMatch] = {}
        for obj in objs:
            scan_id = scan_look_up_cache[obj.scan.id]
            target_id = obj.target.id
            shift_id = mass_shift_cache[obj.mass_shift].id
            inst = cls(
                scan_id=scan_look_up_cache[obj.scan.id],
                is_decoy=is_decoy,
                analysis_id=analysis_id,
                score=obj.score,
                rank=obj.rank,
                q_value=obj.q_value,
                solution_set_id=solution_set_id,
                is_best_match=obj.best_match,
                structure_id=target_id,
                mass_shift_id=shift_id)
            acc.append(inst)
            revmap[scan_id, target_id, shift_id] = obj

        session.bulk_save_objects(acc)

        if save_score_set:
            session.flush()
            fwd = session.query(cls.id, cls.scan_id, cls.structure_id, cls.mass_shift_id).filter(
                cls.solution_set_id == solution_set_id).all()
            acc = []
            loc_acc = []
            for inst in fwd:
                obj = revmap[inst.scan_id, inst.structure_id, inst.mass_shift_id]
                if hasattr(obj, 'score_set'):
                    fields = GlycopeptideSpectrumMatchScoreSet.get_fields_from_object(obj, inst.id)
                    acc.append(fields)
                if obj.localizations:
                    loc_acc.extend(LocalizationScore.get_fields_from_object(obj, inst.id))
            if acc:
                session.bulk_insert_mappings(GlycopeptideSpectrumMatchScoreSet, acc)
            if loc_acc:
                session.bulk_insert_mappings(
                    LocalizationScore,
                    loc_acc
                )

        return revmap

    @classmethod
    def bulk_load(cls, session, ids, chunk_size: int = 512,
                  mass_shift_cache: Optional[Dict] = None,
                  scan_cache: Optional[Dict] = None,
                  structure_cache: Optional[Dict] = None,
                  peptide_relation_cache: Optional[Dict] = None) -> Dict[int, MemorySpectrumMatch]:
        if peptide_relation_cache is None:
            peptide_relation_cache = {}
        if structure_cache is None:
            structure_cache = {}
        if mass_shift_cache is None:
            mass_shift_cache = {m.id: m.convert() for m in session.query(CompoundMassShift).all()}
        if scan_cache is None:
            scan_cache = {}

        out = {}
        id_series = []
        scan_series = []
        mass_shift_series = []
        structure_series = []
        is_best_match_series = Deque()
        rank_series = Deque()
        solution_set_id = Deque()
        score_map = {}

        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(cls.__table__.select().where(
                cls.id.in_(chunk)))
            for self in res.fetchall():
                id_series.append(self.id)
                scan_series.append(self.scan_id)
                mass_shift_series.append(mass_shift_cache[self.mass_shift_id])
                structure_series.append(self.structure_id)
                is_best_match_series.append(self.is_best_match)
                rank_series.append(self.rank)
                solution_set_id.append(self.solution_set_id)
                score_map[self.id] = (self.score, self.q_value)

        solution_set_id_to_cluster_id = {}
        uniq_set_ids = list(set(solution_set_id))
        for chunk in chunkiter(uniq_set_ids, chunk_size):
            res = session.execute(
                select([GlycopeptideSpectrumSolutionSet.id,
                        GlycopeptideSpectrumSolutionSet.cluster_id]).where(
                GlycopeptideSpectrumSolutionSet.id.in_(chunk)))
            for (ssid, clid) in res.fetchall():
                solution_set_id_to_cluster_id[ssid] = clid

        scan_objects = MSScan.bulk_load(
            session, scan_series,
            chunk_size=chunk_size, scan_cache=scan_cache)

        structure_objects = Glycopeptide.bulk_load(
            session, structure_series,
            chunk_size=chunk_size,
            peptide_relation_cache=peptide_relation_cache,
            structure_cache=structure_cache)

        score_sets_map = GlycopeptideSpectrumMatchScoreSet.bulk_load(
            session, ids, chunk_size=chunk_size)

        localization_map = LocalizationScore.bulk_load(session, ids, chunk_size=chunk_size)

        for id, scan, structure, mass_shift, best_match, rank, sset_id in zip(
                id_series,
                scan_objects,
                structure_objects,
                mass_shift_series,
                is_best_match_series,
                rank_series,
                solution_set_id):
            localizations = localization_map[id]
            if id in score_sets_map:
                score_set, fdr_set = score_sets_map[id]
                out[id] = MultiScoreSpectrumMatch(
                    scan, structure,
                    id=id,
                    score_set=score_set,
                    best_match=best_match,
                    q_value_set=fdr_set,
                    mass_shift=mass_shift,
                    localizations=localizations,
                    rank=rank,
                    cluster_id=solution_set_id_to_cluster_id[sset_id]
                )
            else:
                score, q_value = score_map[id]
                out[id] = MemorySpectrumMatch(
                    scan, structure, score, best_match,
                    q_value=q_value,
                    id=id,
                    mass_shift=mass_shift,
                    localizations=localizations,
                    rank=rank,
                    cluster_id=solution_set_id_to_cluster_id[sset_id]
                )
        return out

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
            target = self.structure.convert(peptide_relation_cache=peptide_relation_cache)
        else:
            try:
                target = structure_cache[key]
            except KeyError:
                target = structure_cache[key] = self.structure.convert(peptide_relation_cache=peptide_relation_cache)

        orm_locs = self.localizations
        if orm_locs:
            locs = [loc.convert() for loc in orm_locs]
        else:
            locs = []
        mass_shift = self._resolve_mass_shift(session, mass_shift_cache)
        orm_score_set = self.score_set
        if orm_score_set is None:
            inst = MemorySpectrumMatch(scan, target, self.score, self.is_best_match,
                                       mass_shift=mass_shift, localizations=locs)
        else:
            score_set, fdr_set = orm_score_set.convert()
            inst = MultiScoreSpectrumMatch(
                scan, target, score_set, self.is_best_match,
                mass_shift=mass_shift, q_value_set=fdr_set, match_type=0,
                localizations=locs)

        inst.q_value = self.q_value
        inst.id = self.id
        return inst

    def __repr__(self):
        return "DB" + repr(self.convert())

    @property
    def cluster_id(self):
        return self.solution_set.cluster_id


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
    glycan_coverage = needs_migration(Column(Numeric(8, 7, asdecimal=False), index=False), 0.0)

    total_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    peptide_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    glycan_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)
    glycopeptide_q_value = Column(Numeric(8, 7, asdecimal=False), index=False)

    spectrum_match = relationship(
        GlycopeptideSpectrumMatch, backref=backref("score_set", lazy='joined', uselist=False))

    @classmethod
    def get_fields_from_object(cls, obj, db_id=None):
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
        return fields

    @classmethod
    def serialize_from_spectrum_match(cls, obj, session, db_id):
        fields = cls.get_fields_from_object(obj, db_id)
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

    @classmethod
    def bulk_load(cls, session, ids, chunk_size: int=512) -> Dict[int, Tuple[ScoreSet, FDRSet]]:
        out = {}
        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(cls.__table__.select().where(cls.id.in_(chunk)))
            for self in res.fetchall():
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
                out[self.id] = (score_set, fdr_set)
        return out


class LocalizationScore(Base):
    __tablename__ = "LocalizationScore"

    id = Column(Integer, ForeignKey(
        GlycopeptideSpectrumMatch.id), primary_key=True)

    position = Column(Integer, primary_key=True)
    modification = Column(String)
    score = Column(Numeric(6, 3, asdecimal=False))

    spectrum_match = relationship(
        GlycopeptideSpectrumMatch, backref=backref("localizations", lazy='joined'))

    @classmethod
    def get_fields_from_object(cls, obj: MemorySpectrumMatch, db_id=None):
        instances = []
        for loc in obj.localizations:
            inst = dict(
                id=db_id,
                position=loc.position,
                modification=loc.modification,
                score=loc.score
            )
            instances.append(inst)
        return instances

    def convert(self):
        return MemoryLocalizationScore(self.position, self.modification, self.score)

    def __repr__(self):
        return f"DB{self.__class__.__name__}({self.id}, {self.position}, {self.modification}, {self.score})"

    def __str__(self):
        return f"{self.modification}:{self.position}:{self.score}"

    @classmethod
    def bulk_load(cls, session, ids, chunk_size: int = 512) -> DefaultDict[int, List[MemoryLocalizationScore]]:
        out = DefaultDict(list)
        for chunk in chunkiter(ids, chunk_size):
            res = session.execute(
                cls.__table__.select().where(cls.id.in_(chunk)))
            for self in res.fetchall():
                out[self.id].append(MemoryLocalizationScore(
                    self.position, self.modification, self.score))
        return out
