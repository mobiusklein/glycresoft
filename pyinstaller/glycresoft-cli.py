import matplotlib
import os
import sys
import click
import platform
import multiprocessing

if platform.system().lower() != 'windows':
    os.environ["NOWAL"] = "1"
app_dir = click.get_app_dir("glycresoft")
_mpl_cache_dir = os.path.join(app_dir, 'mpl')

if not os.path.exists(_mpl_cache_dir):
    os.makedirs(_mpl_cache_dir)

os.environ["MPLCONFIGDIR"] = _mpl_cache_dir

try:
    matplotlib.use("agg")
except Exception:
    pass

from rdflib.plugins import stores
from rdflib.plugins.stores import sparqlstore

from glycan_profiling.cli.__main__ import main

from glycan_profiling.cli.validators import strip_site_root

sys.excepthook = strip_site_root

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
