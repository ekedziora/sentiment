from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import movie_reviews
import nltk
import string
from nltk.corpus import stopwords
from nltk.metrics.association import BigramAssocMeasures
from nltk.metrics.scores import precision, recall
from collections import defaultdict
import random

stop = stopwords.words('english')

def normalizeWords(words):
    return [w.lower() for w in words if w.lower() not in stop and w.strip(string.punctuation)]

def findMostFrequentBigrams(words, scoreFunction=BigramAssocMeasures.chi_sq, count=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    return set(bigram_finder.nbest(scoreFunction, count))

def bagOfWordsFeatures(words):
    all_words_normalized = normalizeWords(words)
    most_frequent_bigrams = findMostFrequentBigrams(all_words_normalized, count=2000)

    features = {}
    for unigram in set(all_words_normalized):
        features['contains({})'.format(unigram)] = True
    for bigram in most_frequent_bigrams:
        features['contains({})'.format(bigram)] = True

    return features

def sentimentclassification(trainset, testset):
    (TN, TP, FN, FP) = ( 0 , 0 , 0 , 0)
    classifier = NaiveBayesClassifier.train(trainset)
    for (testReview, polarity) in testset:
        predicted = classifier.classify(testReview)
        if predicted == polarity:
            if predicted == 'neg':
                TN += 1
            else:
                TP += 1
        else:
            if predicted == 'neg':
                FN += 1
            else:
                FP += 1

    return ((TN, TP, FN, FP))

def precision_recall(classifier, testFeatures):
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    for i, (feats, label) in enumerate(testFeatures):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls

featureset = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        features = bagOfWordsFeatures(movie_reviews.words(fileid))
        featureset += [(features, category)]

random.shuffle(featureset)
cutpoint = int(len(featureset) * 0.9)
trainset, testset = featureset[:cutpoint], featureset[cutpoint:]


classifier = nltk.NaiveBayesClassifier.train(trainset)
print("Accurancy: {}".format(nltk.classify.accuracy(classifier, testset)))
precisions, recalls = precision_recall(classifier, testset)
print(precisions)
print(recalls)