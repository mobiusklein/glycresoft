import logging
try:
    import psycopg2
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)
except ImportError:
    pass
logger = logging.getLogger("sqlalchemy.pool.NullPool")
logger.propagate = False
logger.addHandler(logging.NullHandler())

from dill._dill import _reverse_typemap
_reverse_typemap['IntType'] = int
_reverse_typemap['ListType'] = list