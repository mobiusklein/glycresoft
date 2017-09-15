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


pack-pyinstaller:
	cd pyinstaller/dist && 7z a glycresoft-cli.zip glycresoft-cli && cp glycresoft-cli.zip ../


install-dependencies:
	pip install --upgrade pip setuptools wheel
	pip install Cython --install-option="--no-cython-compile"
	pip install lxml pyteomics brain-isotopic-distribution
	pip install --only-binary=numpy,scipy numpy scipy
	pip install -r external-requirements.txt
	python pyinstaller/install-from-git.py
	python setup.py install


update-docs:
	git checkout gh-pages
	git pull origin master
	cd docs && make clean html
	git add docs/_build/html
	git commit -m "update docs"
	git push origin gh-pages
	git checkout master
