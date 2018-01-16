import re
from collections import OrderedDict

from sqlalchemy.ext.baked import bakery
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship, backref, make_transient, Query, validates
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey, PickleType,
    Boolean, Table, Text, Index)
from sqlalchemy.ext.mutable import MutableDict

from glycan_profiling.serialize.base import (
    Base, MutableList)


from .hypothesis import GlycopeptideHypothesis
from .glycan import GlycanCombination

from glycopeptidepy.structure import sequence, residue, PeptideSequenceBase
from glycan_profiling.structure.structure_loader import PeptideProteinRelation, FragmentCachingGlycopeptide


class PeptideSequenceWrapperBase(PeptideSequenceBase):
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


class Protein(Base, PeptideSequenceWrapperBase):
    __tablename__ = "Protein"

    id = Column(Integer, primary_key=True, autoincrement=True)
    protein_sequence = Column(Text, default=u"")
    name = Column(String(128), index=True)
    other = Column(MutableDict.as_mutable(PickleType))
    hypothesis_id = Column(Integer, ForeignKey(
        GlycopeptideHypothesis.id, ondelete="CASCADE"))
    hypothesis = relationship(GlycopeptideHypothesis, backref=backref('proteins', lazy='dynamic'))

    def _get_sequence_str(self):
        return self.protein_sequence

    _n_glycan_sequon_sites = None

    @property
    def n_glycan_sequon_sites(self):
        if self._n_glycan_sequon_sites is None:
            try:
                self._n_glycan_sequon_sites = sequence.find_n_glycosylation_sequons(self.protein_sequence)
            except residue.UnknownAminoAcidException:
                return []
        return self._n_glycan_sequon_sites

    _o_glycan_sequon_sites = None

    @property
    def o_glycan_sequon_sites(self):
        if self._o_glycan_sequon_sites is None:
            try:
                self._o_glycan_sequon_sites = sequence.find_o_glycosylation_sequons(self.protein_sequence)
            except residue.UnknownAminoAcidException:
                return []
        return self._o_glycan_sequon_sites

    _glycosaminoglycan_sequon_sites = None

    @property
    def glycosaminoglycan_sequon_sites(self):
        if self._glycosaminoglycan_sequon_sites is None:
            try:
                self._glycosaminoglycan_sequon_sites = sequence.find_glycosaminoglycan_sequons(self.protein_sequence)
            except residue.UnknownAminoAcidException:
                return []
        return self._glycosaminoglycan_sequon_sites

    @property
    def glycosylation_sites(self):
        try:
            return self.n_glycan_sequon_sites  # + self.o_glycan_sequon_sites
        except residue.UnknownAminoAcidException:
            return []

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


class ProteinSite(Base):
    __tablename__ = "ProteinSite"

    id = Column(Integer, primary_key=True)
    name = Column(String(32), index=True)
    location = Column(Integer, index=True)
    protein_id = Column(Integer, ForeignKey(Protein.id, ondelete="CASCADE"), index=True)


def _convert_class_name_to_collection_name(name):
    parts = re.split(r"([A-Z]+[a-z]+)", name)
    parts = [p.lower() for p in parts if p]
    return '_'.join(parts) + 's'


class PeptideBase(PeptideSequenceWrapperBase):
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


class Peptide(PeptideBase, Base):
    __tablename__ = 'Peptide'

    id = Column(Integer, primary_key=True)

    count_glycosylation_sites = Column(Integer)
    count_missed_cleavages = Column(Integer)
    count_variable_modifications = Column(Integer)

    start_position = Column(Integer)
    end_position = Column(Integer)

    peptide_score = Column(Numeric(12, 6, asdecimal=False))
    peptide_score_type = Column(String(56))

    base_peptide_sequence = Column(String(512))
    modified_peptide_sequence = Column(String(512))

    sequence_length = Column(Integer)

    peptide_modifications = Column(String(128))
    n_glycosylation_sites = Column(MutableList.as_mutable(PickleType))
    o_glycosylation_sites = Column(MutableList.as_mutable(PickleType))
    gagylation_sites = Column(MutableList.as_mutable(PickleType))

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

    __table_args__ = (Index("ix_Peptide_mass_search_index", "calculated_mass", "hypothesis_id"),)


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

    def convert(self):
        inst = FragmentCachingGlycopeptide(self.glycopeptide_sequence)
        inst.id = self.id
        peptide = self.peptide
        inst.protein_relation = PeptideProteinRelation(
            peptide.start_position, peptide.end_position, peptide.protein_id,
            peptide.hypothesis_id)
        return inst

    def __repr__(self):
        return "DBGlycopeptideSequence({self.glycopeptide_sequence}, {self.calculated_mass})".format(self=self)
    _protein_relation = None

    @property
    def protein_relation(self):
        if self._protein_relation is None:
            peptide = self.peptide
            self._protein_relation = PeptideProteinRelation(
                peptide.start_position, peptide.end_position, peptide.protein_id,
                peptide.hypothesis_id)
        return self._protein_relation

    @property
    def glycan_composition(self):
        return self.glycan_combination.convert()

    __table_args__ = (Index("ix_Glycopeptide_mass_search_index", "calculated_mass", "hypothesis_id"),)
