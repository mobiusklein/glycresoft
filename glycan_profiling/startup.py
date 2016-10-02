import warnings
from sqlalchemy import exc as sa_exc
warnings.simplefilter("ignore", category=sa_exc.SAWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    module="pysqlite2.dbapi2")
