from multiprocessing import freeze_support
from glycan_profiling.cli import (
    base, build_db, tools, mzml, analyze, config,
    export)

try:
    from glycresoft_app import server
except ImportError as e:
    pass

main = base.cli.main

if __name__ == '__main__':
    freeze_support()
    main()
