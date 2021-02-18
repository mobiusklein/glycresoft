#!/bin/bash

# NOTE: New versions of PyInstaller bundle more and more hooks, and throw
# errors if there are duplicate hooks. Mangle the names of hooks that have
# been superceded.

# NOTE: PyInstaller and newer versions of Anaconda-specific NumPy are incompatible.
# Even if you're using a conda environment, you must install NumPy with pip in order
# for the executable to work after deactivating the environment or moving it to another
# computer.

rm -rf ./build/ ./dist/
echo "Beginning build"
python -m PyInstaller -c ./glycresoft-cli.py -D \
    -i img/logo.ico \
	--exclude-module _tkinter \
    --exclude-module PyQt4 \
    --exclude-module PyQt5 \
    --exclude-module IPython \
    --exclude-module pandas \
    --workpath build --distpath dist \
    --additional-hooks-dir ./hooks
