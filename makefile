all: test

test:
	# cp glycan_profiling/test/temp_sitecustomize.py ./sitecustomize.py
	# nosetests --with-coverage --cover-package=glycan_profiling --cover-html --cover-html-dir=test_reports\
	# 		 --logging-level=DEBUG -v --with-id glycan_profiling/test/
	# rm ./sitecustomize.py ./sitecustomize.pyc
	py.test -v  glycan_profiling --cov=glycan_profiling --cov-report=html

retest:
	# nosetests --logging-level=DEBUG -v --with-id --failed glycan_profiling/test/
	py.test -v  glycan_profiling --lf	

clean:
	rm ./sitecustomize.py ./sitecustomize.pyc
	coverage erase
