import os
import gzip
import json

from io import BytesIO
from functools import total_ordering

import pickle

from six import string_types as basestring

import sqlalchemy
from sqlalchemy import (
    Column, Integer, String, ForeignKey,
    PickleType, Boolean, Table, ForeignKeyConstraint,
    BLOB)

from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import relationship, deferred
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict

from ms_deisotope.data_source._compression import starts_with_gz_magic

try:
    from ms_deisotope.data_source._compression import starts_with_zstd_magic
except ImportError:
    starts_with_zstd_magic = None

try:
    from pyzstd import ZstdFile
except ImportError:
    ZstdFile = None


from glypy import Composition

from ..base import (
    Base)

from ..utils import get_or_create


def _ipython_key_completions_(self):
    return list(self.keys())


MutableDict._ipython_key_completions_ = _ipython_key_completions_


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
        except Exception:
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


class FileBlob(Base):
    __tablename__ = 'FileBlob'

    id = Column(Integer, primary_key=True)
    name = Column(String(256), index=True)
    data = deferred(Column(BLOB))
    compressed = Column(Boolean)

    def __repr__(self):
        return "FileBlob({self.name})".format(self=self)

    @classmethod
    def from_file(cls, fp, compress=False):
        inst = cls(name=os.path.basename(fp.name), compressed=compress)
        data = fp.read()
        has_gz_magic = starts_with_gz_magic(data)
        if starts_with_zstd_magic is None:
            has_zstd_magic = False
        else:
            has_zstd_magic = starts_with_zstd_magic(data)
        if compress and not has_gz_magic and not has_zstd_magic:
            buff = BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=buff) as compressor:
                compressor.write(data)
            buff.seek(0)
            data = buff.read()
            inst.name += '.gz'
        if not compress and has_gz_magic:
            inst.compressed = True
        if not compress and has_zstd_magic:
            inst.compressed = True
        inst.data = data
        return inst

    @classmethod
    def from_path(cls, path, compress=False):
        with open(path, 'rb') as fh:
            inst = cls.from_file(fh, compress)
        return inst

    def open(self, raw=False):
        data_buffer = BytesIO(self.data)
        if self.compressed and not raw:
            if starts_with_gz_magic(self.data[:6]):
                return gzip.GzipFile(fileobj=data_buffer)
            elif (starts_with_zstd_magic is not None and ZstdFile is not None) and starts_with_zstd_magic(self.data[:6]):
                return ZstdFile(data_buffer)
            else:
                raise ValueError("Could not infer decompressor for %r" % (self.data[:6], ))
        return data_buffer


class HasFiles(object):
    @declared_attr
    def files(cls):
        file_association = Table(
            "%s_Files" % cls.__tablename__,
            cls.metadata,
            Column("file_id", Integer, ForeignKey(FileBlob.id, ondelete="CASCADE"), primary_key=True),
            Column("entity_id", Integer, ForeignKey(
                "%s.id" % cls.__tablename__, ondelete="CASCADE"), primary_key=True))
        cls.FileBlobAssociationTable = file_association
        return relationship(FileBlob, secondary=file_association)

    def add_file(self, file_obj, compress=False):
        if isinstance(file_obj, basestring):
            f = FileBlob.from_path(file_obj, compress)
        else:
            f = FileBlob.from_file(file_obj, compress)
        self.files.append(f)


class JSONType(TypeDecorator):

    impl = sqlalchemy.Text()

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = "J" + json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if isinstance(value, bytes):
                # NOTE: This should only happen when loading a database originally written in Py2?
                if value.startswith(b"J"):
                    value = json.loads(value[1:].decode('utf8'))
                else:
                    value = pickle.loads(value)
            else:
                value = json.loads(value[1:])
        return value


class HasChemicalComposition(object):
    _total_composition = None

    def total_composition(self):
        if self._total_composition is None:
            self._total_composition = Composition(self.formula)
        return self._total_composition
