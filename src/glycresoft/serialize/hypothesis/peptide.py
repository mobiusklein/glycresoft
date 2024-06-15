import re

from collections import OrderedDict

from typing import Any

from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import (
    relationship, backref, Query,
    deferred, object_session)
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Text, Index, select)
from sqlalchemy.ext.mutable import MutableDict, MutableList

from glycresoft.serialize.base import (
    Base)


from .hypothesis import GlycopeptideHypothesis
from .glycan import GlycanCombination
from .generic import JSONType, HasChemicalComposition
from ..utils import chunkiter

from glycopeptidepy.structure import sequence, residue, PeptideSequenceBase
from glycopeptidepy.io.annotation import AnnotationCollection, GlycosylationSite, GlycosylationType
from glycresoft.structure.structure_loader import PeptideProteinRelation, FragmentCachingGlycopeptide
from glycresoft.structure.utils import LRUDict


class AminoAcidSequenceWrapperBase(PeptideSequenceBase):
    _sequence_obj = None

    def _get_sequence_str(self):
        raise NotImplementedError()

    def _get_sequence(self):
        if self._sequence_obj is None:
            self._sequence_obj = sequence.PeptideSequence(
                self._get_sequence_str())
        return self._sequence_obj

    def __iter__(self):
        return iter(self._get_sequence())

    def __getitem__(self, i):
        return self._get_sequence()[i]

    def __len__(self):
        return len(self._get_sequence())

    def __str__(self):
        return str(self._get_sequence())


class Protein(Base, AminoAcidSequenceWrapperBase):
    __tablename__ = "Protein"

    id = Column(Integer, primary_key=True, autoincrement=True)
    protein_sequence = Column(Text, default=u"")
    name = Column(String(128), index=True)
    other = Column(MutableDict.as_mutable(PickleType))
    hypothesis_id = Column(Integer, ForeignKey(
        GlycopeptideHypothesis.id, ondelete="CASCADE"))
    hypothesis = relationship(GlycopeptideHypothesis, backref=backref('proteins', lazy='dynamic'))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        annotations = kwargs.pop("annotations", AnnotationCollection([]))
        self.other = MutableDict()
        super().__init__(*args, **kwargs)
        self.annotations = annotations

    def _get_sequence_str(self):
        return self.protein_sequence.replace("Z", "X")

    _n_glycan_sequon_sites = None

    @property
    def n_glycan_sequon_sites(self):
        if self._n_glycan_sequon_sites is None:
            sites = self.sites.filter(ProteinSite.name == ProteinSite.N_GLYCOSYLATION).all()
            if sites:
                self._n_glycan_sequon_sites = sorted([int(i) for i in sites])
            # else:
            #     try:
            #         self._n_glycan_sequon_sites = sequence.find_n_glycosylation_sequons(self._get_sequence())
            #     except residue.UnknownAminoAcidException:
            #         self._n_glycan_sequon_sites = []
            else:
                self._n_glycan_sequon_sites = []
        return self._n_glycan_sequon_sites

    _o_glycan_sequon_sites = None

    @property
    def o_glycan_sequon_sites(self):
        if self._o_glycan_sequon_sites is None:
            sites = self.sites.filter(ProteinSite.name == ProteinSite.O_GLYCOSYLATION).all()
            if sites:
                self._o_glycan_sequon_sites = sorted([int(i) for i in sites])
            else:
                try:
                    self._o_glycan_sequon_sites = sequence.find_o_glycosylation_sequons(self._get_sequence())
                except residue.UnknownAminoAcidException:
                    self._o_glycan_sequon_sites = []
        return self._o_glycan_sequon_sites

    _glycosaminoglycan_sequon_sites = None

    @property
    def glycosaminoglycan_sequon_sites(self):
        if self._glycosaminoglycan_sequon_sites is None:
            sites = self.sites.filter(ProteinSite.name == ProteinSite.GAGYLATION).all()
            if sites:
                self._glycosaminoglycan_sequon_sites = sorted([int(i) for i in sites])
            else:
                try:
                    self._glycosaminoglycan_sequon_sites = sequence.find_glycosaminoglycan_sequons(
                        self._get_sequence())
                except residue.UnknownAminoAcidException:
                    self._glycosaminoglycan_sequon_sites = []
        return self._glycosaminoglycan_sequon_sites

    @property
    def glycosylation_sites(self):
        try:
            return self.n_glycan_sequon_sites  # + self.o_glycan_sequon_sites
        except residue.UnknownAminoAcidException:
            return []

    def _init_sites(self, include_cysteine_n_glycosylation: bool = False):
        try:
            parsed_sequence = self._get_sequence()
        except residue.UnknownAminoAcidException:
            return
        sites = []
        try:
            n_glycosites = sequence.find_n_glycosylation_sequons(
                parsed_sequence, include_cysteine=include_cysteine_n_glycosylation)
            for n_glycosite in n_glycosites:
                sites.append(
                    ProteinSite(name=ProteinSite.N_GLYCOSYLATION, location=n_glycosite))
        except residue.UnknownAminoAcidException:
            pass
        if self.annotations is not None:
            for modsite in self.annotations.modifications():
                if isinstance(modsite, GlycosylationSite):
                    if modsite.glycosylation_type == GlycosylationType.n_linked:
                        sites.append(
                            ProteinSite(name=ProteinSite.N_GLYCOSYLATION, location=modsite.position)
                        )
                    elif modsite.glycosylation_type == GlycosylationType.o_linked:
                        sites.append(
                            ProteinSite(name=ProteinSite.O_GLYCOSYLATION, location=modsite.position)
                        )
                    elif modsite.glycosylation_type == GlycosylationType.glycosaminoglycan:
                        sites.append(
                            ProteinSite(name=ProteinSite.GAGYLATION, location=modsite.position)
                        )

        # The O- and GAG-linker sites are not determined by a multi AA sequon. We don't
        # need to abstract them away and they are much too common.
        # try:
        #     o_glycosites = sequence.find_o_glycosylation_sequons(
        #         parsed_sequence)
        #     for o_glycosite in o_glycosites:
        #         sites.append(
        #             ProteinSite(name=ProteinSite.O_GLYCOSYLATION, location=o_glycosite))
        # except residue.UnknownAminoAcidException:
        #     pass

        # try:
        #     gag_sites = sequence.find_glycosaminoglycan_sequons(
        #         parsed_sequence)
        #     for gag_site in gag_sites:
        #         sites.append(
        #             ProteinSite(name=ProteinSite.GAGYLATION, location=gag_site))
        # except residue.UnknownAminoAcidException:
        #     pass
        self.sites.extend(sorted(set(sites)))

    def __repr__(self):
        return "DBProtein({0}, {1}, {2}, {3}...)".format(
            self.id, self.name, self.glycosylation_sites,
            self.protein_sequence[:20] if self.protein_sequence is not None else "")

    def to_json(self, full=False):
        d = OrderedDict((
            ('id', self.id),
            ('name', self.name),
            ("glycosylation_sites", list(self.glycosylation_sites)),
            ('other', self.other)
        ))
        if full:
            d.update({
                "protein_sequence": self.protein_sequence
            })
            for k, v in self.__dict__.items():
                if isinstance(v, Query):
                    d[k + '_count'] = v.count()
        return d

    def reverse(self, copy_id=False, prefix=None, suffix=None):
        n = len(self.protein_sequence)
        sites = []
        for site in self.sites:  # pylint: disable=access-member-before-definition
            sites.append(site.__class__(name=site.name, location=n - site.location - 1))
        name = self.name
        if name.startswith(">"):
            if prefix:
                name = ">" + prefix + name[1:]
        if suffix:
            name = name + suffix

        inst = self.__class__(name=name, protein_sequence=self.protein_sequence[::-1])
        if copy_id:
            inst.id = self.id
        inst.sites = sites
        return inst

    @property
    def annotations(self) -> AnnotationCollection:
        return self.other.get('annotations', AnnotationCollection([]))

    @annotations.setter
    def annotations(self, value):
        if not isinstance(value, AnnotationCollection) and value is not None:
            value = AnnotationCollection(value)
        elif value is None:
            value = AnnotationCollection([])
        self.other['annotations'] = value


class ProteinSite(Base):
    __tablename__ = "ProteinSite"

    id = Column(Integer, primary_key=True)
    name = Column(String(32), index=True)
    location = Column(Integer, index=True)
    protein_id = Column(Integer, ForeignKey(Protein.id, ondelete="CASCADE"), index=True)
    protein = relationship(Protein, backref=backref("sites", lazy='dynamic'))

    N_GLYCOSYLATION = "N-Glycosylation"
    O_GLYCOSYLATION = "O-Glycosylation"
    GAGYLATION = "Glycosaminoglycosylation"

    _hash = None

    def __repr__(self):
        return ("{self.__class__.__name__}(location={self.location}, "
                "name={self.name})").format(self=self)

    def __hash__(self):
        hash_val = self._hash
        if hash_val is None:
            hash_val = self._hash = hash((self.name, self.location))
        return hash_val

    def __index__(self):
        return self.location

    def __int__(self):
        return self.location

    def __add__(self, other):
        return int(self) + int(other)

    def __radd__(self, other):
        return int(self) + int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __eq__(self, other):
        if isinstance(other, ProteinSite):
            return self.location == other.location and self.name == other.name
        return int(self) == int(other)

    def __ne__(self, other):
        return not (self == other)


def _convert_class_name_to_collection_name(name):
    parts = re.split(r"([A-Z]+[a-z]+)", name)
    parts = [p.lower() for p in parts if p]
    return '_'.join(parts) + 's'


class PeptideBase(AminoAcidSequenceWrapperBase):
    @declared_attr
    def protein_id(self):
        return Column(Integer, ForeignKey(
            Protein.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def hypothesis_id(self):
        return Column(Integer, ForeignKey(
            GlycopeptideHypothesis.id, ondelete="CASCADE"), index=True)

    @declared_attr
    def protein(self):
        if not hasattr(self, "__collection_name__"):
            name = _convert_class_name_to_collection_name(self.__name__)
        else:
            name = self.__collection_name__
        return relationship(Protein, backref=backref(name, lazy='dynamic'))

    calculated_mass = Column(Numeric(12, 6, asdecimal=False), index=True)
    formula = Column(String(128))

    def __iter__(self):
        return iter(self.convert())

    def __len__(self):
        return len(self.convert())

    @property
    def total_mass(self):
        return self.convert().total_mass

    _protein_relation = None

    @property
    def protein_relation(self):
        if self._protein_relation is None:
            peptide = self
            self._protein_relation = PeptideProteinRelation(
                peptide.start_position, peptide.end_position, peptide.protein_id,
                peptide.hypothesis_id)
        return self._protein_relation

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


class Peptide(PeptideBase, HasChemicalComposition, Base):
    __tablename__ = 'Peptide'

    id = Column(Integer, primary_key=True)

    count_glycosylation_sites = Column(Integer)
    count_missed_cleavages = Column(Integer)
    count_variable_modifications = Column(Integer)

    start_position = Column(Integer)
    end_position = Column(Integer)

    peptide_score = Column(Numeric(12, 6, asdecimal=False))
    _scores = deferred(Column("scores", JSONType))
    peptide_score_type = Column(String(56))

    base_peptide_sequence = Column(String(512))
    modified_peptide_sequence = Column(String(512))

    sequence_length = Column(Integer)

    n_glycosylation_sites = Column(MutableList.as_mutable(JSONType))
    o_glycosylation_sites = Column(MutableList.as_mutable(JSONType))
    gagylation_sites = Column(MutableList.as_mutable(JSONType))

    hypothesis = relationship(GlycopeptideHypothesis, backref=backref('peptides', lazy='dynamic'))

    def _get_sequence_str(self):
        return self.modified_peptide_sequence

    def convert(self):
        inst = sequence.parse(self.modified_peptide_sequence)
        inst.id = self.id
        return inst

    def __repr__(self):
        return ("DBPeptideSequence({self.modified_peptide_sequence}, {self.n_glycosylation_sites},"
                " {self.start_position}, {self.end_position})").format(self=self)

    __table_args__ = (
        Index("ix_Peptide_mass_search_index", "hypothesis_id", "calculated_mass"),
        Index("ix_Peptide_coordinate_index", "id", "calculated_mass",
              "start_position", "end_position"),)

    @hybrid_method
    def spans(self, position):
        return position in self.protein_relation

    @spans.expression
    def spans(self, position):
        # Overhang at the end is for consistency with SpanningMixin, but this is not
        # ideal
        return (self.start_position <= position) & (position <= self.end_position)

    @hybrid_property
    def scores(self): # pylint: disable=method-hidden
        try:
            if self._scores is None:
                self.scores = []
            return self._scores
        except Exception:
            return []

    @scores.setter
    def scores(self, value):
        self._scores = value


class Glycopeptide(PeptideBase, Base):
    __tablename__ = "Glycopeptide"

    id = Column(Integer, primary_key=True)
    peptide_id = Column(Integer, ForeignKey(Peptide.id, ondelete='CASCADE'), index=True)
    glycan_combination_id = Column(Integer, ForeignKey(GlycanCombination.id, ondelete='CASCADE'), index=True)

    peptide = relationship(Peptide, backref=backref("glycopeptides", lazy='dynamic'))
    glycan_combination = relationship(GlycanCombination)

    glycopeptide_sequence = Column(String(1024))

    hypothesis = relationship(GlycopeptideHypothesis, backref=backref('glycopeptides', lazy='dynamic'))

    def _get_sequence_str(self):
        return self.glycopeptide_sequence

    @classmethod
    def bulk_load(cls, session, ids, chunk_size: int=512, peptide_relation_cache=None, structure_cache=None):
        if peptide_relation_cache is None:
            peptide_relation_cache = session.info.get("peptide_relation_cache")
            if peptide_relation_cache is None:
                peptide_relation_cache = session.info['peptide_relation_cache'] = LRUDict(
                    maxsize=1024)
        if structure_cache is None:
            structure_cache = {}
        out = structure_cache
        peptide_ids = []
        for chunk in chunkiter(ids, chunk_size):
            chunk = list(filter(lambda x: x not in structure_cache, chunk))
            res = session.execute(cls.__table__.select().where(
                cls.id.in_(chunk)
                )
            )

            for self in res.fetchall():
                inst = FragmentCachingGlycopeptide(self.glycopeptide_sequence)
                inst.id = self.id

                peptide_id = self.peptide_id
                out[self.id] = (inst, peptide_id)
                if peptide_id not in peptide_relation_cache:
                    peptide_ids.append(peptide_id)

        for chunk in chunkiter(peptide_ids, chunk_size):
            res = session.execute(Peptide.__table__.select().where(
                Peptide.id.in_(
                    list(filter(lambda x: x not in peptide_relation_cache,
                                chunk)))))

            for self in res.fetchall():
                peptide_relation_cache[self.id] = PeptideProteinRelation(
                    self.start_position,
                    self.end_position,
                    self.protein_id,
                    self.hypothesis_id
                )

        result = []
        for i in ids:
            inst, peptide_id = out[i]
            inst.protein_relation = peptide_relation_cache[peptide_id]
            result.append(inst)
        return result

    def convert(self, peptide_relation_cache=None):
        if peptide_relation_cache is None:
            session = object_session(self)
            peptide_relation_cache = session.info.get("peptide_relation_cache")
            if peptide_relation_cache is None:
                peptide_relation_cache = session.info['peptide_relation_cache'] = LRUDict(maxsize=1024)
        inst = FragmentCachingGlycopeptide(self.glycopeptide_sequence)
        inst.id = self.id
        peptide_id = self.peptide_id
        if peptide_id in peptide_relation_cache:
            inst.protein_relation = peptide_relation_cache[self.peptide_id]
        else:
            session = object_session(self)
            peptide_props = session.query(
                Peptide.start_position, Peptide.end_position,
                Peptide.protein_id, Peptide.hypothesis_id).filter(Peptide.id == peptide_id).first()
            peptide_relation_cache[peptide_id] = inst.protein_relation = PeptideProteinRelation(*peptide_props)
        return inst

    def __repr__(self):
        return "DBGlycopeptideSequence({self.glycopeptide_sequence}, {self.calculated_mass})".format(self=self)
    _protein_relation = None

    @hybrid_method
    def is_multiply_glycosylated(self):
        return self.glycan_combination.count > 1

    @is_multiply_glycosylated.expression
    def is_multiply_glycosylated(self):
        expr = select([GlycanCombination.count > 1]).where(
            GlycanCombination.id == Glycopeptide.glycan_combination_id).label(
            "is_multiply_glycosylated")
        return expr

    @property
    def protein_relation(self):
        if self._protein_relation is None:
            peptide = self.peptide
            self._protein_relation = PeptideProteinRelation(
                peptide.start_position, peptide.end_position, peptide.protein_id,
                peptide.hypothesis_id)
        return self._protein_relation

    @property
    def n_glycan_sequon_sites(self):
        return self.peptide.n_glycosylation_sites

    @property
    def o_glycan_sequon_sites(self):
        return self.peptide.o_glycosylation_sites

    @property
    def glycosaminoglycan_sequon_sites(self):
        return self.peptide.gagylation_sites

    @property
    def glycan_composition(self):
        return self.glycan_combination.convert()

    __table_args__ = (
        Index("ix_Glycopeptide_mass_search_index", "hypothesis_id", "calculated_mass"),
        # Index("ix_Glycopeptide_mass_search_index_full", "calculated_mass", "hypothesis_id",
        #                                                 "peptide_id", "glycan_combination_id"),
    )
