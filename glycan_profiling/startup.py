import dill  # Registers converters to allow more types to be pickled
import warnings
from sqlalchemy import exc as sa_exc

from glycan_profiling.config.config_file import get_configuration

warnings.simplefilter("ignore", category=sa_exc.SAWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", module="SPARQLWrapper")
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    module="pysqlite2.dbapi2")


# http://stackoverflow.com/a/15472811/1137920
def _setup_win32_keyboard_interrupt_handler():
    import os
    from scipy import stats
    import thread
    import threading
    import win32api

    def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
        if dwCtrlType == 0:
            hook_sigint()
            print("Keyboard Interrupt", threading.current_thread(), len(threading.enumerate()))
            print(threading.enumerate())
            return 1
        return 0

    win32api.SetConsoleCtrlHandler(handler, 1)


get_configuration()

try:
    _setup_win32_keyboard_interrupt_handler()
except ImportError:
    pass
