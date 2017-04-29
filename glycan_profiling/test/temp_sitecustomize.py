import os
try:
    import coverage

    os.environ['COVERAGE_PROCESS_START'] = os.path.join(os.path.dirname(__file__), ".coveragerc")
    coverage.process_startup()
except ImportError:
    pass
