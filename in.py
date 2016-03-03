from nltk.collocations import BigramCollocationFinder
from nltk.corpus import movie_reviews
import nltk
import random
import string
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.metrics.association import BigramAssocMeasures

stop = stopwords.words('english')

def normalizeWords(words):
    return [w.lower() for w in words if w.lower() not in stop and w.strip(string.punctuation)]


unigrams_categorized = [(set(normalizeWords(movie_reviews.words(fileid))), category)
                        for category in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(category)]

bigrams_categorized = [(set(bigrams(words)), category)
                       for (words, category) in unigrams_categorized]

random.shuffle(unigrams_categorized)
random.shuffle(bigrams_categorized)

all_words_normalized = normalizeWords(movie_reviews.words())

all_words_dist = nltk.FreqDist(all_words_normalized)
most_frequent_unigrams = list(all_words_dist)[:2000]

bigram_finder = BigramCollocationFinder.from_words(all_words_normalized)
most_frequent_bigrams = bigram_finder.nbest(BigramAssocMeasures.student_t, 2000)
# all_bigrams_dist = nltk.FreqDist(bigrams(all_words_normalized))
# most_frequent_bigrams = list(all_bigrams_dist)[:2000]

def features_bigrams(document):
    doc_bigrams = set(document)
    features = {}
    for b in most_frequent_bigrams:
        features['contains({})'.format(b)] = (b in doc_bigrams)
    return features

def features_unigrams(document):
    doc_unigrams = set(document)
    features = {}
    for unigram in most_frequent_unigrams:
        features['contains({})'.format(unigram)] = (unigram in doc_unigrams)
    return features

# featureset = [(features_bigrams(bigram), category) for (bigram, category) in bigrams_categorized]
featureset = [(features_unigrams(unigram), category) for (unigram, category) in unigrams_categorized]
cutpoint = int(len(featureset) * 0.9)
trainset, testset = featureset[:cutpoint], featureset[cutpoint:]


classifier = nltk.NaiveBayesClassifier.train(trainset)
classifier.show_most_informative_features(20)
print(nltk.classify.accuracy(classifier, testset))