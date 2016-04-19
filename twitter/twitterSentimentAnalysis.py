from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
from utils import findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, createWordsInCategoriesDictionary, performCrossValidation
from featureExtractors import *
from normalization import normalizeTwitterWords, normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt',
#                                                cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# testCorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
corpus = CategorizedPlaintextCorpusReader('corpus/twitter-data/manualsSO', r'(\w+)-tweet[0-9]+\.txt',
                                                cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

featureset = []
labels = []
normalizationFunction = normalizeTwitterWords

allWordsNormalized = normalizationFunction(corpus.words())
wordsInCategories = createWordsInCategoriesDictionary(corpus, normalizationFunction)
bestWordsCount = int(len(allWordsNormalized) * 0.2)
bestWords = findBestWords(wordsInCategories, max_words=bestWordsCount)
bestBigrams = findMostFrequentBigrams(allWordsNormalized, count=bestWordsCount)
bestTrigrams = findMostFrequentTrigrams(allWordsNormalized, count=bestWordsCount)
mpqaDictionary = MpqaDictionaryWrapper()

i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        features = {}
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        wordsTagged = nltk.pos_tag(normalizedWords)
        features = unigramsFeatures(normalizedWords, bestWords)
        features.update(bigramsFeatures(normalizedWords, bestBigrams))
        features.update(trigramsFeatures(normalizedWords, bestTrigrams))
        features.update(sentiwordnetSentimentWordsCountFeatures(wordsTagged))
        features.update(mpqaSubjectivityWordsScoreFeatures(wordsTagged, mpqaDictionary))
        features.update(posTagsCountFeatures(wordsTagged))
        features.update(mpqaObjectivityWordsScoreFeatures(wordsTagged, mpqaDictionary))
        features.update(extraTwitterFeaturesPresence(extraNormalizedWords))
        featureset += [(features, category)]
        labels.append(category)
        print(i)
        i += 1

performCrossValidation(featureset, labels, 10, False)