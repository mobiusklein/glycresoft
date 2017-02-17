import dill  # Registers converters to allow more types to be pickled
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
