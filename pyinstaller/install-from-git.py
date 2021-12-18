from os import path as ospath
import sys
import os
import shutil

repos = [
    "https://github.com/mobiusklein/pysqlite.git",
    "https://github.com/mobiusklein/glypy.git",
    "https://github.com/mobiusklein/glycopeptidepy.git",
    "https://github.com/mobiusklein/ms_deisotope.git",
]

clone_dir = ospath.join(ospath.dirname(__file__), "gitsrc")

origin_path = os.getcwd()
os.system("rm -rf %s" % clone_dir)

for repo in repos:
    repopath = ospath.join(clone_dir, ospath.splitext(ospath.basename(repo))[0])
    os.system("pip install git+%s" % repo)
