import os

newFolder = r'manualsSO'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

file = open("Manually-Annotated-Tweets.tsv", encoding='utf8')


def getSentimentAsString(sentiment):
    if sentiment == 'positive' or sentiment == 'negative':
        return 'sub'
    elif sentiment == 'objnspam' or sentiment == 'objspam':
        return 'obj'



for index, line in enumerate(file.read().splitlines()):
    if line.startswith(" "):
        splitted = line.rsplit("\t", 1)
        sentiment = getSentimentAsString(splitted[1])
        tweetId = index
        tweetContent = splitted[0].strip()
        file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+', encoding='utf8')
        file.write(tweetContent)
        file.close()