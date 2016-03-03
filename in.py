from nltk.classify.naivebayes import NaiveBayesClassifier
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

def sentimentclassification(trainset, testset):
    (TN, TP, FN, FP) = ( 0 , 0 , 0 , 0)
    classifier = NaiveBayesClassifier.train(trainset)
    for (testReview, polarity) in testset:
        predicted = classifier.classify(testReview)
        if predicted == polarity:
            if predicted == 'neg':
                TN += 1
            else:
                TP += 1
        else:
            if predicted == 'neg':
                FN += 1
            else:
                FP += 1

    return ((TN, TP, FN, FP))

featureset = []

for fileid in movie_reviews.fileids():
    all_words_normalized = normalizeWords(movie_reviews.words(fileid))
    category = movie_reviews.categories(fileid)[0]

    bigram_finder = BigramCollocationFinder.from_words(all_words_normalized)
    most_frequent_bigrams = set(bigram_finder.nbest(BigramAssocMeasures.student_t, 1000))

    features = {}
    for unigram in set(all_words_normalized):
        features['contains({})'.format(unigram)] = True
    for b in most_frequent_bigrams:
        features['contains({})'.format(b)] = True

    featureset += [(features, category)]


cutpoint = int(len(featureset) * 0.9)
trainset, testset = featureset[:cutpoint], featureset[cutpoint:]


classifier = nltk.NaiveBayesClassifier.train(trainset)
classifier.show_most_informative_features(20)
print(nltk.classify.accuracy(classifier, testset))