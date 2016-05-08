import os

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import pickle
from featureExtractors import *
from lexicons.mpqa.mpqaDictionary import MpqaDictionaryWrapper
from normalization import normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle
from utils import findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, createWordsInCategoriesDictionary, performCrossValidation, performTestValidation

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC, LinearSVR, NuSVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/train', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
testcorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
corpus = CategorizedPlaintextCorpusReader('corpus/3-way/datacopy', r'(\w+)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)


def getfeatures(normalizedWords, extraNormalizedWords = list()):
    features = {}
    # wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    # features.update(trigramsFeatures(normalizedWords))
    # features.update(posTagsCountFeatures(wordsTagged))
    # features.update(sentiwordnetSentimentWordsCountFeatures(wordsTagged))
    # features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaObjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(extraTwitterFeaturesCount(extraNormalizedWords))
    return features

def getfeaturesTest(normalizedWords, extraNormalizedWords):
    features = {}
    wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    # features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(extraTwitterFeaturesCount(extraNormalizedWords))
    return features

def getFeaturesetFromPickle():
    with open('wordsTaggedToCategory-3way', 'rb') as pickFile:
        wordsTaggedWithCategory = pickle.load(pickFile)

    featureset = []
    for wordsTagged, category in wordsTaggedWithCategory:
        features = {}
        # features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
        features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
        featureset += [(features, category)]

    return featureset

featureset = []
labels = []
normalizationFunction = normalizeTwitterWordsWithNegationHandle

mpqaDictionary = MpqaDictionaryWrapper()

i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        features = getfeatures(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        featureset += [(features, category)]
        labels.append(category)
        i += 1

print(i)


for i, featuresPlusCategory in enumerate(getFeaturesetFromPickle()):
    initFeatures, initCategory = featureset[i]
    features, category = featuresPlusCategory
    initFeatures.update(features)

sklearclassifier = LinearSVC()
performCrossValidation(featureset, labels, 10, sklearclassifier)

testfeatureset = []

classifier = SklearnClassifier(sklearclassifier).train(featureset)

# newFolder = r'dumps/3way/multiNB'
# newFolder = r'dumps/3way/logreg'
newFolder = r'dumps/3way/linsvc'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

# with open(newFolder + "/uni", 'wb+') as fileout:
# with open(newFolder + "/uni-bi", 'wb+') as fileout:
# with open(newFolder + "/uni-bi-extra", 'wb+') as fileout:
# with open(newFolder + "/uni-bi-extra-mpqa-senti", 'wb+') as fileout:
with open(newFolder + "/uni-bi-extra-mpqa-subj", 'wb+') as fileout:
    pickle.dump(classifier, fileout)

for category in testcorpus.categories():
    for fileid in testcorpus.fileids(category):
        words = testcorpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        testfeatures = getfeaturesTest(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        testfeatureset += [(testfeatures, category)]

performTestValidation(featureset, testfeatureset, sklearclassifier)