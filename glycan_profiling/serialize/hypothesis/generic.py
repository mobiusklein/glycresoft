from functools import total_ordering
from sqlalchemy import (
    Column, Numeric, Integer, String, ForeignKey,
    PickleType, Boolean, Table, ForeignKeyConstraint)

from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict


from ms_deisotope.output.db import (
    Base)

from ..utils import get_or_create


@total_ordering
class ApplicationVersion(Base):
    __tablename__ = "ApplicationVersion"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    major = Column(Integer)
    minor = Column(Integer)
    patch = Column(Integer)

    def __eq__(self, other):
        if other is None:
            return False
        try:
            if self.name != other.name:
                return False
        except AttributeError:
            pass

        return tuple(self) == tuple(other)

    def __lt__(self, other):
        if other is None:
            return False
        try:
            if self.name != other.name:
                return False
        except AttributeError:
            pass

        return tuple(self) < tuple(other)

    def __iter__(self):
        yield self.major
        yield self.minor
        yield self.patch


class ParameterStore(object):
    parameters = Column(MutableDict.as_mutable(PickleType), default=dict)


class Taxon(Base):
    __tablename__ = "Taxon"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), index=True)

    @classmethod
    def get(cls, session, id, name=None):
        obj, made = get_or_create(session, cls, id=id, name=name)
        return obj

    def __repr__(self):
        return "<Taxon {} {}>".format(self.id, self.name)


class HasTaxonomy(object):

    @declared_attr
    def taxa(cls):
        taxon_association = Table(
            "%s_Taxa" % cls.__tablename__,
            cls.metadata,
            Column("taxon_id", Integer, ForeignKey(Taxon.id, ondelete="CASCADE"), primary_key=True),
            Column("entity_id", Integer, ForeignKey(
                "%s.id" % cls.__tablename__, ondelete="CASCADE"), primary_key=True))
        cls.TaxonomyAssociationTable = taxon_association
        return relationship(Taxon, secondary=taxon_association)

    @classmethod
    def with_taxa(cls, ids):
        try:
            iter(ids)
            return cls.taxa.any(Taxon.id.in_(tuple(ids)))
        except:
            return cls.taxa.any(Taxon.id == ids)


class ReferenceDatabase(Base):
    __tablename__ = "ReferenceDatabase"
    id = Column(Integer, primary_key=True)
    name = Column(String(128))
    url = Column(String(128))

    @classmethod
    def get(cls, session, id=None, name=None, url=None):
        obj, made = get_or_create(session, cls, id=id, name=name, url=url)
        return obj

    def __repr__(self):
        return "<ReferenceDatabase {} {}>".format(self.id, self.name)


class ReferenceAccessionNumber(Base):
    __tablename__ = "ReferenceAccessionNumber"
    id = Column(String(64), primary_key=True)
    database_id = Column(Integer, ForeignKey(ReferenceDatabase.id), primary_key=True)
    database = relationship(ReferenceDatabase)

    @classmethod
    def get(cls, session, id, database_id):
        obj, made = get_or_create(session, cls, id=id, database_id=database_id)
        return obj

    def __repr__(self):
        return "<ReferenceAccessionNumber {} {}>".format(self.id, self.database.name)


class HasReferenceAccessionNumber(object):
    @declared_attr
    def references(cls):
        reference_number_association = Table(
            "%s_ReferenceAccessionNumber" % cls.__tablename__,
            cls.metadata,
            Column("accession_code", String(64), primary_key=True),
            Column("database_id", Integer, primary_key=True),
            Column(
                "entity_id", Integer, ForeignKey("%s.id" % cls.__tablename__, ondelete="CASCADE"), primary_key=True),
            ForeignKeyConstraint(
                ["accession_code", "database_id"],
                ["ReferenceAccessionNumber.id", "ReferenceAccessionNumber.database_id"]))
        cls.ReferenceAccessionAssocationTable = reference_number_association
        return relationship(ReferenceAccessionNumber, secondary=reference_number_association)


TemplateNumberStore = Table("TemplateNumberStore", Base.metadata, Column("value", Integer))
