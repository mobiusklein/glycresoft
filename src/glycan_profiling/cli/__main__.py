import os
import sys
import traceback

from multiprocessing import freeze_support

import click

from glycan_profiling.cli import (
    base, build_db, tools, mzml, analyze, config,
    export)

try:
    from glycresoft_app.cli import server
except ImportError as e:
    pass


def info(type, value, tb):
    if not sys.stderr.isatty():
        click.secho("Running interactively, not starting debugger", fg='yellow')
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)


def set_breakpoint_hook():
    try:
        import ipdb
        sys.breakpointhook = ipdb.set_trace
    except ImportError:
        pass


def main():
    freeze_support()
    if os.getenv("GLYCRESOFTDEBUG"):
        sys.excepthook = info
        click.secho("Running glycresoft with debugger", fg='yellow')
    if os.getenv("GLYCRESOFTPROFILING"):
        import cProfile
        click.secho("Running glycresoft with profiler", fg='yellow')
        profiler = cProfile.Profile()
        profiler.runcall(base.cli.main, standalone_mode=False)
        profiler.dump_stats('glycresoft_performance.profile')
    else:
        base.cli.main(standalone_mode=True)


if __name__ == '__main__':
    main()
