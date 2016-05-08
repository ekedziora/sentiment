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

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
testcorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

def performTestValidation(testset, polarClassifierName, sentiClassifierName):
        with open(polarClassifierName, 'rb') as fileout:
            polarClassifier = pickle.load(fileout)
        with open(sentiClassifierName, 'rb') as fileout:
            sentiClassifier = pickle.load(fileout)


        precisions, recalls, accuracy = precision_recall_2step(polarClassifier, sentiClassifier, testset)

        print("Test accuracy: {0:.3f}".format(accuracy))
        precRecall = {label: (precision, recalls.get(label)) for label, precision in precisions.items()}
        for label, (prec, recall) in precRecall.items():
            print("Precision for {0}: {1:.3f}".format(label, prec))
            print("Recall for {0}: {1:.3f}".format(label, recall))
            fmeasure = 2 * prec * recall/(prec + recall)
            print("F measure for {0}: {1:.3f}".format(label, fmeasure))


def getfeaturesTest(normalizedWords, extraNormalizedWords):
    features = {}
    wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    # features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
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
        testfeatureset += [(testfeatures, category)]

# performTestValidation(testfeatureset, "dumps/2step/polar/multiNB/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/multiNB/uni-bi-extra-mpqa-subj")
# performTestValidation(testfeatureset, "dumps/2step/polar/multiNB/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/logreg/uni-bi-extra-mpqa-subj")
# performTestValidation(testfeatureset, "dumps/2step/polar/logreg/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/multiNB/uni-bi-extra-mpqa-subj")
performTestValidation(testfeatureset, "dumps/2step/polar/logreg/uni-bi-extra-mpqa-subj", "dumps/2step/sentiment/logreg/uni-bi-extra-mpqa-subj")