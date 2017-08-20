all: test

test:
	py.test -v  glycan_profiling --cov=glycan_profiling --cov-report=html -s

retest:
	py.test -v  glycan_profiling --lf	

clean:
	rm -rf build/ dist/ pyinstaller/build/ pyinstaller/dist pyinstaller/gitsrc/
	coverage erase
	

build-pyinstaller:
	cd pyinstaller && bash make-pyinstaller.sh
	pyinstaller/dist/glycresoft-cli/glycresoft-cli -h
