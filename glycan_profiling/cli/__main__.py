from glycan_profiling.cli import (
    base, build_db, inspect, mzml, analyze)

try:
    from glycresoft_app import server
except ImportError, e:
    print(e)

main = base.cli.main

if __name__ == '__main__':
    main()
