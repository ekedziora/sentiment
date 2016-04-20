import csv, os

newFolders = [r'test']

for newFolder in newFolders:
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)

csvFile = open("testdata.manual.2009.06.14.csv")
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
    sentiment = getSentimentAsStringFromInt(row[0])
    tweetId = int(row[1])
    tweetContent = row[5]
    file = open(r'test/{}-tweet{}.txt'.format(sentiment, tweetId), 'w+')
    file.write(tweetContent)