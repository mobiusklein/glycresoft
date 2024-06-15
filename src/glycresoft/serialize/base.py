try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import (
    Column, Numeric, String)
from sqlalchemy.orm import validates
from sqlalchemy.orm.session import object_session

Base = declarative_base()


def Mass(index=True):
    return Column(Numeric(14, 6, asdecimal=False), index=index)


def find_by_name(session, model_class, name):
    return session.query(model_class).filter(model_class.name == name).first()


def make_unique_name(session, model_class, name):
    marked_name = name
    i = 1
    while find_by_name(session, model_class, marked_name) is not None:
        marked_name = "%s (%d)" % (name, i)
        i += 1
    return marked_name


class HasUniqueName(object):
    name = Column(String(128), default=u"", unique=True)
    uuid = Column(String(64), index=True, unique=True)

    @classmethod
    def make_unique_name(cls, session, name):
        return make_unique_name(session, cls, name)

    @classmethod
    def find_by_name(cls, session, name):
        return find_by_name(session, cls, name)

    @validates("name")
    def ensure_unique_name(self, key, name):
        session = object_session(self)
        if session is not None:
            model_class = self.__class__
            name = make_unique_name(session, model_class, name)
            return name
        else:
            return name
