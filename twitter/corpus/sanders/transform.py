import csv, os

newFolder = r'data'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

csvFile = open("full-corpus.csv", encoding='utf8')
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
        if not tweetContent:
            continue
        file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+', encoding='utf8')
        file.write(tweetContent)