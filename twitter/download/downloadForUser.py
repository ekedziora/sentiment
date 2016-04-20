import os, tweepy

CONSUMER_KEY = 'aMs4H5LwxiUDOg1l7w6GTavAS'
CONSUMER_SECRET = 'kLOlkVXX1zmuYdMUHDsnRXSfG6PZQKjHSJWUgvvUAUUSjvvYXz'
ACCESS_KEY = '718512636523081729-AzY9a6MuTBcFbHE61qkA4Jn4RBUg5Nn'
ACCESS_SECRET = 'wJKpJEpyP7qZSNTqQzZU3wIhuqWb1KPdrIN1jTUz09rs1'

newFolder = r'data'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

users = ['washingtonpost', 'nytimes', 'nypost']

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)

api = tweepy.API(auth)

globalCount = 0

for user in users:
    allTweets = []

    tweets = api.user_timeline(screen_name=user, count=200)
    globalCount += len(tweets)
    allTweets.extend(tweets)
    oldest = allTweets[-1].id - 1

    while len(tweets) > 0:
        print("Getting tweets of user: {} before tweetId: {}".format(user, oldest))
        tweets = api.user_timeline(screen_name=user, count=200, max_id=oldest)
        globalCount += len(tweets)
        allTweets.extend(tweets)
        oldest = allTweets[-1].id - 1
        print("Number of user tweets downloaded: {}".format(len(allTweets)))

    for tweet in allTweets:
        id = tweet.id
        with open(r'{}/neu-tweet{}.txt'.format(newFolder, id), 'w') as outfile:
            outfile.write(tweet.text)

    print("Global tweets downloaded: {}".format(globalCount))