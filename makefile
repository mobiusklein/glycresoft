all: test


test:
	cp glycan_profiling/test/temp_sitecustomize.py ./sitecustomize.py
	nosetests --with-coverage --with-timer --cover-package=glycan_profiling --cover-html --cover-html-dir=test_reports --logging-level=DEBUG -v --with-id glycan_profiling/test/
	rm ./sitecustomize.py

retest:
	nosetests --cover-package=glycan_profiling --logging-level=DEBUG -v --with-id --failed glycan_profiling/test/

clean:
	echo "no-op"
