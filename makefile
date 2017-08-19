all: test

test:
	py.test -v  glycan_profiling --cov=glycan_profiling --cov-report=html -s

retest:
	py.test -v  glycan_profiling --lf	

build-pyinstaller:
	cd pyinstaller && bash make-pyinstaller.sh
	ls pyinstaller/dist/glycresoft-cli/*
	pyinstaller/dist/glycresoft-cli/glycresoft-cli -h
