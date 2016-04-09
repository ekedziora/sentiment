from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
from utils import bagOfWordsFeatures, findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, \
    createWordsInCategoriesDictionary, performCrossValidation, bestUnigramsWithPosFeatures, bestUnigramsWithPosFeaturesTaggedForNormalizedWords
from featureExtractors import *
from normalization import normalizeTwitterWords, normalizeTwitterWordsWithExtraFeatures

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
trainCorpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt',
                                               cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# testCorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

featureset = []
labels = []
normalizationFunction = normalizeTwitterWords

allWordsNormalized = normalizationFunction(trainCorpus.words())
wordsInCategories = createWordsInCategoriesDictionary(trainCorpus, normalizationFunction)
bestWords = findBestWords(wordsInCategories, max_words=4000)
# bestBigrams = findMostFrequentBigrams(allWordsNormalized, count=30000)
# bestTrigrams = findMostFrequentTrigrams(allWordsNormalized, count=30000)

print("best unigramy + pos presence")
i = 1
for category in trainCorpus.categories():
    for fileid in trainCorpus.fileids(category):
        words = trainCorpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        features = {}
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        wordsTagged = nltk.pos_tag(normalizedWords)
        features = unigramsFeatures(normalizedWords, bestWords)
        # features.update(bigramsFeatures(normalizedWords, bestBigrams))
        # features.update(trigramsFeatures(normalizedWords, bestTrigrams))
        # features.update(sentiwordnetSentimentWordsCountFeatures(wordsTagged))
        features.update(posTagsPresenceFeatrues(wordsTagged))
        # features.update(extraTwitterFeaturesPresence(extraNormalizedWords))
        featureset += [(features, category)]
        labels.append(category)
        print(i)
        i += 1

performCrossValidation(featureset, labels, 10, False)