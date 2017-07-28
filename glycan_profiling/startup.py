'''
This module sets up any preconditions for using this library, either
programmatically or as a command line program, for both manager and
worker processes.

It is the first component imported by glycan_profiling/__init__.py
which means any components it imports may in turn import from
glycan_profiling/__init__.py could result in a circular import
problem.
'''

# Registers converters to allow more types to be pickled
import dill
import warnings
from sqlalchemy import exc as sa_exc
from glycan_profiling.config.config_file import get_configuration

warnings.simplefilter("ignore", category=sa_exc.SAWarning)
warnings.filterwarnings(action="ignore", module="SPARQLWrapper")
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    module="pysqlite2.dbapi2")


get_configuration()
