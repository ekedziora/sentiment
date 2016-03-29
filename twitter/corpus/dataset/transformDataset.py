import csv, os

newFolders = [r'pos', r'neg', r'neu']

for newFolder in newFolders:
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)

csvFile = open("Sentiment Analysis Dataset.csv", encoding='utf8')
reader = csv.reader(csvFile)


def getSentimentAsStringFromInt(sentiment):
    sentimentNumber = int(sentiment)
    if sentimentNumber == 0:
        return 'neg'
    elif sentimentNumber == 2:
        return 'neu'
    elif sentimentNumber == 4:
        return 'pos'



for index, row in enumerate(reader):
    if index in range(30000):
        sentiment = getSentimentAsStringFromInt(row[0])
        tweetId = int(row[1])
        tweetContent = row[5]
        file = open(r'train/{}/tweet{}.txt'.format(sentiment, tweetId), 'w+', encoding='utf8')
        file.write(tweetContent)