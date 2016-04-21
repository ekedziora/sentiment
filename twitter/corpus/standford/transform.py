import csv, os

newFolder = r'trainAll'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

csvFile = open("training.1600000.processed.noemoticon.csv")
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
	if index in range(1000000, 1200000):
		sentiment = getSentimentAsStringFromInt(row[0])
		tweetId = int(row[1])
		tweetContent = row[5]
		file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+')
		file.write(tweetContent)
		
	if index % 200 == 0:
		print("Processed {} tweets".format(index))