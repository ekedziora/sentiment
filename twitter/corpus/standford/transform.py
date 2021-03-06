import csv, os

def crawl(reader):
    for index, row in enumerate(reader):
        if index in range(0,300000):
            sentiment = getSentimentAsStringFromInt(row[0])
            tweetId = int(row[1])
            tweetContent = row[5]
            if tweetContent:
                file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+', encoding='utf8')
                file.write(tweetContent)
            if index % 200 == 0:
                print("Processed {} tweets".format(index))

def getSentimentAsStringFromInt(sentiment):
    sentimentNumber = int(sentiment)
    if sentimentNumber == 0:
        return 'neg'
    elif sentimentNumber == 2:
        return 'neu'
    elif sentimentNumber == 4:
        return 'pos'

newFolder = r'trainAll'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

csvFile = open("training.1600000.processed.noemoticon_pos.csv", encoding='cp1252')
reader = csv.reader(csvFile)

crawl(reader)

csvFile = open("training.1600000.processed.noemoticon_neg.csv", encoding='cp1252')
reader = csv.reader(csvFile)

crawl(reader)
