import shutil
from os import listdir
from os.path import isfile, join

frompath = 'datacopy'
topath = 'polar'

onlyfiles = [f for f in listdir(frompath) if isfile(join(frompath, f))]

print(len(onlyfiles))
i = 0
j = 0
for f in onlyfiles:
    start = f[0:3]
    if start == 'neu':
        shutil.copyfile(frompath + '/' + f, topath + '/notpolar-tweet' + str(i) + ".txt")
        i += 1
    else:
        shutil.copyfile(frompath + '/' + f, topath + '/polar-tweet' + str(j) + ".txt")
        j += 1