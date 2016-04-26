import os, shutil
from os import listdir
from os.path import isfile, join
from collections import defaultdict

mypath = 'datacopy'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

s = defaultdict(int)

for f in onlyfiles:
    start = f[0:3]
    s[start] += 1

frompath = '../twitter-data/autos'

autosfiles = [f for f in listdir(frompath) if isfile(join(frompath, f))]

for f in autosfiles:
    type = f[0:3]
    if s[type] < s['neu']:
        if not os.path.exists(mypath + "/" + f) and not os.stat(frompath + "/" + f).st_size == 0:
            shutil.copyfile(frompath + '/' + f, mypath + "/" + f)
            s[type] += 1