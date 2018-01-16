from os import path as ospath
import sys
import os
import shutil

repos = [
    "https://github.com/mobiusklein/pysqlite.git",
    "https://github.com/mobiusklein/glypy.git",
    "https://github.com/mobiusklein/glycopeptidepy.git",
    "https://github.com/mobiusklein/ms_peak_picker.git",
    "https://github.com/mobiusklein/brainpy.git",
    "https://github.com/mobiusklein/psims.git",
    "https://github.com/mobiusklein/ms_deisotope.git",
]

clone_dir = ospath.join(ospath.dirname(__file__), "gitsrc")

origin_path = os.getcwd()
os.system("rm -rf %s" % clone_dir)

for repo in repos:
    repopath = ospath.join(clone_dir, ospath.splitext(ospath.basename(repo))[0])
    os.system("git clone %s %s" % (repo, repopath))
    os.chdir(repopath)
    os.system("\"%s\" setup.py install" % (sys.executable,))
    os.chdir(origin_path)
