import os

newFolder = r'autos'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

file = open("Auto-Annotated-Negative-Tweets.txt", encoding='utf8')

for index, line in enumerate(file.read().splitlines()):
    tweetId = index
    tweetContent = line
    file = open(r'{}/{}-tweet{}.txt'.format(newFolder, 'neg', tweetId), 'w+', encoding='utf8')
    file.write(tweetContent)
    file.close()

file = open("Auto-Annotated-Positive-Tweets.txt", encoding='utf8')

for index, line in enumerate(file.read().splitlines()):
    tweetId = index
    tweetContent = line
    if not tweetContent:
        continue
    file = open(r'{}/{}-tweet{}.txt'.format(newFolder, 'pos', tweetId), 'w+', encoding='utf8')
    file.write(tweetContent)
    file.close()