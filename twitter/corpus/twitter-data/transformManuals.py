import os

newFolder = r'manuals'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

file = open("Manually-Annotated-Tweets.tsv", encoding='utf8')


def getSentimentAsString(sentiment):
    if sentiment == 'positive':
        return 'pos'
    elif sentiment == 'negative':
        return 'neg'
    elif sentiment == 'objnspam' or sentiment == 'objspam':
        return 'neu'



for index, line in enumerate(file.read().splitlines()):
    if line.startswith(" "):
        splitted = line.rsplit("\t", 1)
        sentiment = getSentimentAsString(splitted[1])
        tweetId = index
        tweetContent = splitted[0].strip()
        if not tweetContent:
            continue
        file = open(r'{}/{}-tweet{}.txt'.format(newFolder, sentiment, tweetId), 'w+', encoding='utf8')
        file.write(tweetContent)
        file.close()