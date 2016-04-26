import shutil, os

from os import listdir
from os.path import isfile, join
from collections import defaultdict

mypath = 'datacopy'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
    if os.stat(mypath + "/" + f).st_size == 0:
        os.remove(mypath + "/" + f)