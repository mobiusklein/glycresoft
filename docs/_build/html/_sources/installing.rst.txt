Installation
------------

To install :title-reference:`glycresoft`, you will need git, Python 2.7,
a C compiler compatible with that Python version, Make, and the ability To
install NumPy and SciPy. It is recommended that you create a virtual
environment for the purpose.

1. git clone https://github.com/mobiusklein/glycresoft.git
2. cd glycresoft
3. make install-dependencies

At this point the command line tool ``glycresoft`` will be available, and should
be on your ``PATH`` tied to the current Python environment.

To build a bundled executable not tied to the active Python environment, follow these
steps:

4. pip install PyInstaller
5. make build-pyinstaller
6. ./pyinstall/dist/glycresoft-cli/glycresoft-cli -h

Now, the compiled artefacts in ./pyinstaller/dist/glycresoft-cli can be relocated
and ran by anyone without needing to have a version of Python configured and
installed.
