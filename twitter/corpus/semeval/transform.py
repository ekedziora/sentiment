import os

newFolder = r'data'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

file = open("traindata")


def getSentimentAsString(sentiment):
    if sentiment == 'positive':
        return 'pos'
    elif sentiment == 'negative':
        return 'neg'
    elif sentiment == 'objective-OR-neutral' or sentiment == 'objective' or sentiment == 'neutral':
        return 'neu'
    else:
        return None



for index, line in enumerate(file.read().splitlines()):
    splitted = line.split("\t")
    sentiment = getSentimentAsString(splitted[2].strip('"'))
    tweetId = splitted[1]
    tweetContent = splitted[3].strip()
    if os.path.exists(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId)) or not tweetContent:
        continue
    file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+', encoding='utf8')
    file.write(tweetContent)
    file.close()