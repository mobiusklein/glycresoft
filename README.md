GlycReSoft
----------
Software for analyzing glycomics and glycoproteomics LC-MS/MS data

Usage
=====

This package provides several command line tools, available under the name `glycresoft`.

```
$ glycresoft -h
Commands:
  analyze           Identify structures in preprocessed data
  build-hypothesis  Build search spaces for glycans and glycopeptides
  config            Set persistent configuration options
  export            Write Data Collections To Text Files
  mzml              Inspect and preprocess mzML files
  tools             Odds and ends to help inspect data and diagnose issues
```

Installing
==========

This program requires Python 3.8 or newer, last tested with Python 3.10. The simplest way to perform the installation is to use a virtual environment to isolate its dependencies. To install from source after cloning this repository:

```bash
pip install -v .
```
or alternatively to install certain dependencies from source
```bash
make install-dependencies
```

This will install this library and all of its dependencies into the current Python environment. The command line tool will be made available under the name `glycresoft`.

If you want to build the standalone executable that bundles all of its dependencies with its own version of Python, next run

```bash
pip install PyInstaller
make build-pyinstaller
```

This will install PyInstaller, a Python program which can package another Python program and all of its dependencies into a native executable, and run it on an entry point wrapper for GlycReSoft. Afterwards, the standalone executable and its static files are located in `./pyinstaller/dist/glycresoft-cli/`, where the executable is named `glycresoft-cli`, which will operate essentially the same as the `glycresoft` executable installed above, save that it will do some platform specific configuration and can be used from outside the virtual environment it was created in.

When installing pre-compiled wheels from the package index, installation should take less than 5 minutes on a regular computer. Installing packages from source things may take up to 15 minutes.

Usage
=====

For a brief tutorial on the original high collision energy glycopeptide search tool, see the [Original Tutorial](https://mobiusklein.github.io/glycresoft/docs/_build/html/tutorials/glycoproteomics-tutorial.html).

For a full tutorial on the stepped collision energy glycopeptide search tool, see