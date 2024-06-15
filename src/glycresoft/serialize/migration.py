
from typing import Type
from sqlalchemy.engine import Connection

from glycresoft.task import log_handle

from .base import Base
from .utils import has_column


class Migration:
    connection: Connection
    tp: Type[Base]

    def __init__(self, connection: Connection, tp: Type[Base]):
        self.connection = connection
        self.tp = tp

    def add_column(self, prop):
        log_handle.log(f"Adding column {prop} for {self.tp}")
        column_name = str(prop.compile(dialect=self.connection.dialect)).split('.')[-1]
        column_type = prop.type.compile(self.connection.dialect)
        migration = prop.info['needs_migration']
        if migration['default'] is None:
            self.connection.execute(
                f'ALTER TABLE {self.tp.__tablename__} ADD COLUMN {column_name} {column_type};')
        else:
            self.connection.execute(
                f'ALTER TABLE {self.tp.__tablename__} ADD COLUMN {column_name} {column_type} DEFAULT {migration["default"]};')

    def needs_migration(self):
        for prop in self.tp.__mapper__.columns:
            if prop.info.get("needs_migration") and not has_column(self.connection, self.tp.__tablename__, prop.name):
                self.add_column(prop)

    def run(self):
        self.needs_migration()
