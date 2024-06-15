all: test


dev:
	pip install -e . --no-build-isolation -v


test:
	py.test -v  tests --cov=glycresoft --cov-report=html -s -l -ra


retest:
	py.test -v  tests --lf -l -ra


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
	pip install Cython
	pip install lxml pyteomics brain-isotopic-distribution
	pip install --only-binary=numpy numpy
	pip install --only-binary=scipy "scipy"
	pip install -r external-requirements.txt
	python pyinstaller/install-from-git.py
	python setup.py install


update-docs:
	git checkout gh-pages
	git pull origin master
	cd docs && make clean html
	git add -f docs/_build/html
	git commit -m "update docs"
	git push origin gh-pages
	git checkout master
