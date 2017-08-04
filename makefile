all: test

test:
	py.test -v  glycan_profiling --cov=glycan_profiling --cov-report=html -s

retest:
	py.test -v  glycan_profiling --lf	
