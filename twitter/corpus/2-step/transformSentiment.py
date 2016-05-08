import shutil
from os import listdir
from os.path import isfile, join

frompath = 'datacopy'
topath = 'sentiment'

onlyfiles = [f for f in listdir(frompath) if isfile(join(frompath, f))]

for f in onlyfiles:
    start = f[0:3]
    if start != 'neu':
        shutil.copyfile(frompath + '/' + f, topath + '/' + f)