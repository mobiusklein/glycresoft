Installation
------------

To install :title-reference:`glycresoft`, you will need git, Python 3.8, 3.9, or 3.10,
a C compiler compatible with that Python version. It is recommended that you create a virtual
environment for the purpose.

Installing from PyPI
======================

`glycresoft <https://pypi.org/project/glycresoft/>`_ can be installed from the Python Package Index (PyPI).

.. code-block:: bash
    :linenos:

    pip install -v glycresoft



Installing From Source
=======================

.. code-block:: bash
    :linenos:

    git clone https://github.com/mobiusklein/glycresoft.git
    cd glycresoft
    make install-dependencies

At this point the command line tool ``glycresoft`` will be available, and should
be on your ``$PATH`` tied to the current Python environment.

For PyInstaller
================

To build a bundled executable not tied to the active Python environment, follow these
steps:

.. code-block:: bash
    :linenos:

    pip install PyInstaller
    cd ./pyinstaller/
    bash make-pyinstaller.sh
    ./dist/glycresoft-cli/glycresoft-cli -h

Now, the compiled artefacts in ``pyinstaller/dist/glycresoft-cli`` can be relocated
and ran by anyone without needing to have a version of Python configured and
installed.
