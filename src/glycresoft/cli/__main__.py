import os
import sys
import traceback
import platform
import logging

from multiprocessing import freeze_support, set_start_method

import click

from glycresoft.cli import (
    base, build_db, tools, mzml, analyze, config,
    export)

try:
    from glycresoft_app.cli import server
except ImportError:
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
    try:
        if platform.system() == 'Windows' or platform.system() == "Darwin":
            set_start_method("spawn")
        else:
            set_start_method("forkserver")
    except Exception as err:
        logging.getLogger().error("Failed to set multiprocessing start method: %s", err, exc_info=True)

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
