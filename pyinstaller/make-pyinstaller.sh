#!/bin/bash

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
    --additional-hooks-dir ./
