from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer

from normalization import normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle
import pickle, nltk

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
corpus = CategorizedPlaintextCorpusReader('corpus/2-step/polar', r'(\w+)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)

normalizationFunction = normalizeTwitterWordsWithNegationHandle

wordsTaggedToCategory = []

i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        wordsTagged = nltk.pos_tag(normalizedWords)
        wordsTaggedToCategory += [(wordsTagged, category)]
        print(i)
        i += 1

with open("wordsTaggedToCategory-polar", 'wb') as fileout:
    pickle.dump(wordsTaggedToCategory, fileout)