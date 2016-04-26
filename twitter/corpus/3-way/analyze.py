import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict

mypath = 'datacopy'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

s = defaultdict(int)

for f in onlyfiles:
    if os.stat(mypath + "/" + f).st_size == 0:
        print(f)
    start = f[0:3]
    s[start] += 1

print(s)