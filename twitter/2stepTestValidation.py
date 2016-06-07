from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC, LinearSVR, NuSVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron

from featureExtractors import unigramsFeatures, bigramsFeatures, mpqaSubjectivityWordsCountFeatures, \
    extraTwitterFeaturesCount, mpqaSentimentWordsCountFeatures
from lexicons.mpqa.mpqaDictionary import MpqaDictionaryWrapper
from normalization import normalizeTwitterWordsWithNegationHandle, normalizeTwitterWordsWithExtraFeatures
from utils import precision_recall_2step

import nltk, pickle
import csv
from itertools import product

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
testcorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

def performTestValidation(testset, polarClassifierName, sentiClassifierName):
        with open(polarClassifierName, 'rb') as fileout:
            polarClassifier = pickle.load(fileout)
        with open(sentiClassifierName, 'rb') as fileout:
            sentiClassifier = pickle.load(fileout)

        labels = ['pos', 'neg', 'neu']
        precisions, recalls, fscores, accuracy = precision_recall_2step(polarClassifier, sentiClassifier, testset, labels)

        print("Test accuracy: {0:.3f}".format(accuracy))
        measures = {label: (precision, recall, fscore) for label, precision, recall, fscore in zip(labels, precisions, recalls, fscores)}
        for label, (prec, recall, fscore) in measures.items():
            print("Precision for {0}: {1:.3f}".format(label, prec))
            print("Recall for {0}: {1:.3f}".format(label, recall))
            print("F measure for {0}: {1:.3f}".format(label, fscore))
        return fscores, accuracy


def getfeaturesTest(normalizedWords, extraNormalizedWords):
    features = {}
    wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(extraTwitterFeaturesCount(extraNormalizedWords))
    return features


mpqaDictionary = MpqaDictionaryWrapper()

normalizationFunction = normalizeTwitterWordsWithNegationHandle

testfeatureset = []

for category in testcorpus.categories():
    for fileid in testcorpus.fileids(category):
        words = testcorpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        testfeatures = getfeaturesTest(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        testfeatureset.append((testfeatures, category))

# performTestValidation(testfeatureset, "dumps/2step/polar/multiNB/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/multiNB/uni-bi-extra-mpqa-subj")
# performTestValidation(testfeatureset, "dumps/2step/polar/multiNB/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/logreg/uni-bi-extra-mpqa-subj")
# performTestValidation(testfeatureset, "dumps/2step/polar/logreg/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/multiNB/uni-bi-extra-mpqa-subj")
# performTestValidation(testfeatureset, "dumps/2step/polar/logreg/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/logreg/uni-bi-extra-mpqa-subj")

with open("results2.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(('features', 'polar classifier', 'sentiment classifier', 'test accuracy', 'f score for pos', 'f score for neg', 'f score for neu'))

    polarDir = "dumps/2step/polar/"
    sentimentDir = "dumps/2step/sentiment"
    versionsSet = ['multiNB/60pct', 'multiNB/70pct', 'multiNB/80pct', 'logreg/60pct', 'logreg/70pct', 'logreg/80pct']
    for featuresVersion in ['uni-bi-extra-mpqa-senti']:

        tuples = product(versionsSet, versionsSet)
        for tuple in tuples:
            # print("CLASSIFIERS:")
            # print("Polar: " + tuple[0])
            # print("Sentiment: " + tuple[1])
            polarClassifierPath = polarDir + '/' + tuple[0] + '/' + featuresVersion
            sentimentClassifierPath = sentimentDir + '/' + tuple[1] + '/' + featuresVersion
            fscores, accuracy = performTestValidation(testfeatureset, polarClassifierPath, sentimentClassifierPath)
            csvwriter.writerow((featuresVersion, tuple[0], tuple[1], accuracy, fscores[0], fscores[1], fscores[2]))
            # print("\n\n")