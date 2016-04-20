import csv, os

newFolder = r'data'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

csvFile = open("full-corpus.csv")
reader = csv.reader(csvFile)


def getSentimentAsString(sentiment):
    if sentiment == 'positive':
        return 'pos'
    elif sentiment == 'negative':
        return 'neg'
    elif sentiment == 'neutral':
        return 'neu'
    else:
        return None


for index, row in enumerate(reader):
    sentiment = getSentimentAsString(row[1])
    if sentiment is not None:
        tweetId = int(row[2])
        tweetContent = row[4]
        file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+')
        file.write(tweetContent)