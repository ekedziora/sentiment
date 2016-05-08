from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer

from normalization import normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle
import pickle, nltk

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
corpus = CategorizedPlaintextCorpusReader('corpus/2-step/sentiment', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

normalizationFunction = normalizeTwitterWordsWithNegationHandle

wordsTaggedToCategory = []
labels = []

i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        wordsTagged = nltk.pos_tag(normalizedWords)
        wordsTaggedToCategory += [(wordsTagged, category)]
        labels.append(category)
        print(i)
        i += 1

with open("wordsTaggedToCategory-sentiment", 'wb') as fileout:
    pickle.dump(wordsTaggedToCategory, fileout)

with open("labels-sentiment", 'wb') as fileout:
    pickle.dump(labels, fileout)