from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer

from lexicons.mpqa.mpqaDictionary import MpqaDictionaryWrapper
from utils import findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, createWordsInCategoriesDictionary, performCrossValidation
from featureExtractors import *
from normalization import normalizeTwitterWords, normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/trainAll', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
testcorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
corpus = CategorizedPlaintextCorpusReader('corpus/3-way/datacopy', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)


def getfeatures(normalizedWords, extraNormalizedWords = list(), bestWords = set(), bestBigrams = set()):
    features = {}
    # wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords, bestWords))
    features.update(bigramsFeatures(normalizedWords))
    features.update(trigramsFeatures(normalizedWords))
    features.update(extraTwitterFeaturesPresence(extraNormalizedWords))
    return features

featureset = []
labels = []
normalizationFunction = normalizeTwitterWordsWithNegationHandle

mpqaDictionary = MpqaDictionaryWrapper()

allWordsNormalized = normalizationFunction(corpus.words())
wordsInCategories = createWordsInCategoriesDictionary(corpus, normalizationFunction)
bestWordsCount = int(len(allWordsNormalized) * 0.2)
bestWords = findBestWords(wordsInCategories, max_words=bestWordsCount)
bestBigrams = findMostFrequentBigrams(allWordsNormalized, count=bestWordsCount)
# bestTrigrams = findMostFrequentTrigrams(allWordsNormalized, count=bestWordsCount)


i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        features = getfeatures(normalizedWords, extraNormalizedWords=extraNormalizedWords, bestBigrams=bestBigrams)
        featureset += [(features, category)]
        labels.append(category)
        print(i)
        i += 1

performCrossValidation(featureset, labels, 10, False)

testfeatureset = []

for category in testcorpus.categories():
    for fileid in testcorpus.fileids(category):
        words = testcorpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        testfeatures = getfeatures(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        testfeatureset += [(testfeatures, category)]


classifier = nltk.NaiveBayesClassifier.train(featureset)
acc = nltk.classify.accuracy(classifier, testfeatureset)

print("Test accurancy: {}".format(acc))