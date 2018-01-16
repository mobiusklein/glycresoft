import matplotlib
import os
import sys
import click
import platform
import multiprocessing

if platform.system().lower() != 'windows':
    os.environ["NOWAL"] = "1"
else:
    # while click.echo works when run under normal conditions,
    # it has started failing when packaged with PyInstaller. The
    # implementation of click's _winterm module seems to replicate
    # a lot of logic found in win_unicode_console.streams, but using
    # win_unicode_console seems to fix the problem, (found after tracing
    # why importing ipdb which imported IPython which called this fixed
    # the problem)
    import win_unicode_console
    win_unicode_console.enable()
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
