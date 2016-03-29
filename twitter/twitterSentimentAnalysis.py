from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
from utils import bagOfWordsFeatures, findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, \
    createWordsInCategoriesDictionary, performCrossValidation, bestUnigramsWithPosFeatures, bestUnigramsWithPosFeaturesTaggedForNormalizedWords
from stemmingLemmingUtils import doStemming, doWordnetLemmatization
from nltk.collocations import BigramAssocMeasures
from normalization import normalizeWords
import re
from featureExtractors import *
from normalization import normalizeTwitterWords, normalizeTwitterWordsWithNegationHandle

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=True)
trainCorpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt',
                                               cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# testCorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

featureset = []
labels = []
normalizationFunction = normalizeTwitterWords

allWordsNormalized = normalizationFunction(trainCorpus.words())
wordsInCategories = createWordsInCategoriesDictionary(trainCorpus, normalizationFunction)
# bestWords = findBestWords(wordsInCategories, max_words=4000)
# bestBigrams = findMostFrequentBigrams(allWordsNormalized, count=30000)
# bestTrigrams = findMostFrequentTrigrams(allWordsNormalized, count=30000)

i = 1
for category in trainCorpus.categories():
    for fileid in trainCorpus.fileids(category):
        normalizedWords = normalizationFunction(trainCorpus.words(fileids=[fileid]))
        features = posTagsCountFeatures(normalizedWords)
        # features.update(unigramsFeatures(ww, bestWords=bestWords))
        # features.update(bigramsFeatures(ww))
        # features.update(trigramsFeatures(ww, bestTrigrams=bestTrigrams))
        featureset += [(features, category)]
        labels.append(category)
        print(i)
        i += 1

for features, category in featureset:
    for word, flag in features.items():
        if word.startswith("neg_"):
            print(features)

performCrossValidation(featureset, labels, 10, False)