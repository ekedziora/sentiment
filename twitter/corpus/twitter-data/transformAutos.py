import os

newFolder = r'autos'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

file = open("Auto-Annotated-Negative-Tweets.txt")

for index, line in enumerate(file.read().splitlines()):
    tweetId = index
    tweetContent = line
    file = open(r'{}/{}-tweet{}.txt'.format(newFolder, 'neg', tweetId), 'w+')
    file.write(tweetContent)
    file.close()

file = open("Auto-Annotated-Positive-Tweets.txt")

for index, line in enumerate(file.read().splitlines()):
    tweetId = index
    tweetContent = line
    file = open(r'{}/{}-tweet{}.txt'.format(newFolder, 'pos', tweetId), 'w+')
    file.write(tweetContent)
    file.close()