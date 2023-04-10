import os
from typing import Callable, Union, Protocol, runtime_checkable

from six import string_types as basestring

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.engine import Connectable

from sqlalchemy import exc
from sqlalchemy.engine import Engine
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import Query

from .base import Base
from .migration import Migration
from .tandem import GlycopeptideSpectrumMatch, GlycopeptideSpectrumMatchScoreSet


ConnectFrom = Union[str,
                    Connectable,
                    'ConnectionRecipe',
                    scoped_session,
                    Session]


@runtime_checkable
class HasSession(Protocol):
    session: Union[Session, scoped_session]


def configure_connection(connection: ConnectFrom, create_tables=True):
    if isinstance(connection, basestring):
        try:
            eng = create_engine(connection)
        except exc.ArgumentError:
            eng = SQLiteConnectionRecipe(connection)()
    elif isinstance(connection, Connectable):
        eng = connection
    elif isinstance(connection, ConnectionRecipe):
        eng = connection()
    elif isinstance(connection, (scoped_session, Session)):
        eng = connection.get_bind()
    elif isinstance(connection, HasSession):
        eng = connection.session.get_bind()
        create_tables = False
    else:
        raise ValueError(
            "Could not determine how to get a database connection from %r" % connection)
    if create_tables:
        Base.metadata.create_all(bind=eng)
        Migration(eng, GlycopeptideSpectrumMatch).run()
        Migration(eng, GlycopeptideSpectrumMatchScoreSet).run()
    return eng


initialize = configure_connection

def _noop_on_connect(connection, connection_record):
    pass


class ConnectionRecipe(object):
    connection_url: str
    connect_args: dict
    on_connect: Callable
    engine_args: dict

    def __init__(self, connection_url: str, connect_args=None, on_connect=None, **engine_args):
        if connect_args is None:
            connect_args = {}
        if on_connect is None:
            on_connect = _noop_on_connect

        self.connection_url = connection_url
        self.connect_args = connect_args
        self.on_connect = on_connect
        self.engine_args = engine_args

    def __call__(self) -> Engine:
        connection = create_engine(
            self.connection_url, connect_args=self.connect_args,
            **self.engine_args)
        event.listens_for(connection, 'connect')(self.on_connect)
        return connection


class SQLiteConnectionRecipe(ConnectionRecipe):
    connect_args = {
        'timeout': 180,
    }

    @staticmethod
    def _configure_connection(dbapi_connection, connection_record):
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before any DDL.
        iso_level = dbapi_connection.isolation_level
        dbapi_connection.isolation_level = None
        try:
            dbapi_connection.execute("PRAGMA page_size = 5120;")
            dbapi_connection.execute("PRAGMA cache_size = 12000;")
            dbapi_connection.execute("PRAGMA temp_store = 3;")
            if not int(os.environ.get("NOWAL", '0')):
                dbapi_connection.execute("PRAGMA journal_mode = WAL;")
                dbapi_connection.execute("PRAGMA wal_autocheckpoint = 100;")
            # dbapi_connection.execute("PRAGMA foreign_keys = ON;")
            dbapi_connection.execute("PRAGMA journal_size_limit = 1000000;")
            pass
        except Exception as e:
            print(e)
        dbapi_connection.isolation_level = iso_level

    def __init__(self, connection_url, **engine_args):
        super(SQLiteConnectionRecipe, self).__init__(
            self._construct_url(connection_url), self.connect_args, self._configure_connection)

    def _construct_url(self, path):
        if path.startswith("sqlite://"):
            return path
        else:
            return "sqlite:///%s" % path


class DatabaseBoundOperation(object):
    engine: Engine
    _original_connection: ConnectFrom
    _sessionmaker: scoped_session

    def __init__(self, connection):
        self.engine = self._configure_connection(connection)

        self._original_connection = connection
        self._sessionmaker = scoped_session(
            sessionmaker(bind=self.engine, autoflush=False))

    def bridge(self, other: 'DatabaseBoundOperation'):
        self.engine = other.engine
        self._sessionmaker = other._sessionmaker

    def _configure_connection(self, connection: ConnectFrom):
        eng = configure_connection(connection, create_tables=True)
        return eng

    def __eq__(self, other):
        return str(self.engine) == str(other.engine) and\
            (self.__class__ is other.__class__)

    def __hash__(self):
        return hash(str(self.engine))

    def __ne__(self, other):
        return not (self == other)

    def __reduce__(self):
        return self.__class__, (self._original_connection,)

    @property
    def session(self):
        return self._sessionmaker

    def query(self, *args) -> Query:
        return self.session.query(*args)

    def _analyze_database(self):
        conn = self.engine.connect()
        inner = conn.connection.connection
        cur = inner.cursor()
        cur.execute("analyze;")
        inner.commit()

    def _sqlite_reload_analysis_plan(self):
        conn = self.engine.connect()
        conn.execute("ANALYZE sqlite_master;")

    def _sqlite_checkpoint_wal(self):
        conn = self.engine.connect()
        inner = conn.connection.connection
        cur = inner.cursor()
        cur.execute("PRAGMA wal_checkpoint(SQLITE_CHECKPOINT_RESTART);")

    @property
    def dialect(self):
        return self.engine.dialect

    def is_sqlite(self):
        return self.dialect.name == "sqlite"

    def close(self):
        self.session.remove()
        self.engine.dispose()
