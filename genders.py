from nltk.corpus import names as n
import random
import nltk

def gender_features(word):
    return {'last_letter': word[-1], 'length': len(word), 'first_letter': word[0]}

males = [('male', name) for name in n.words("male.txt")]
females = [('female', name) for name in n.words("female.txt")]
all = males + females
random.shuffle(all)

features = [(gender_features(w), gender) for (gender, w) in all]
split = int(len(features) * 0.8)
trainset = features[:split]
testset = features[split:]
classifier = nltk.NaiveBayesClassifier.train(trainset)

print(classifier.classify(gender_features('')))