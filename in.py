from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import movie_reviews
import nltk
import string
from nltk.corpus import stopwords
from nltk.metrics.association import BigramAssocMeasures
from nltk.metrics.scores import precision, recall
from collections import defaultdict
from sklearn import cross_validation
from nltk.stem.porter import PorterStemmer

stop = stopwords.words('english')
stemmer = PorterStemmer()

def normalizeWords(words):
    return [stemmer.stem(w.lower()) for w in words if w.lower() not in stop and w.strip(string.punctuation)]

def findMostFrequentBigrams(words, scoreFunction=BigramAssocMeasures.chi_sq, count=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    return set(bigram_finder.nbest(scoreFunction, count))

def bagOfWordsFeatures(words):
    all_words_normalized = normalizeWords(words)
    most_frequent_bigrams = findMostFrequentBigrams(all_words_normalized, count=2000)

    features = {}
    for unigram in set(all_words_normalized):
        features['contains({})'.format(unigram)] = True
    for bigram in most_frequent_bigrams:
        features['contains({})'.format(bigram)] = True

    return features

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

def precision_recall(classifier, testFeatures):
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    for i, (feats, label) in enumerate(testFeatures):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls

featureset = []
labels = []
foldsCount = 10

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        features = bagOfWordsFeatures(movie_reviews.words(fileid))
        featureset += [(features, category)]
        labels.append(category)

cv_iter = cross_validation.StratifiedKFold(labels, n_folds=foldsCount)

i = 1
accurancySum = 0.0
precisionSums = defaultdict(float)
recallSums = defaultdict(float)
for train, test in cv_iter:
    trainset = [featureset[i] for i in train]
    testset = [featureset[i] for i in test]
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    print("Fold {}:".format(i))
    i += 1
    accurancy = nltk.classify.accuracy(classifier, testset)
    accurancySum += accurancy
    print("Accurancy: {}".format(accurancy))
    precisions, recalls = precision_recall(classifier, testset)
    for label, value in precisions.items():
        print("Precision for {}: {}".format(label, value))
        precisionSums[label] += value
    for label, value in recalls.items():
        print("Recall for {}: {}".format(label, value))
        recallSums[label] += value

print("Average accurancy: {}".format(accurancySum/foldsCount))
for label, sum in precisionSums.items():
    print("Average precision for {}: {}".format(label, sum/foldsCount))
for label, sum in recallSums.items():
    print("Average recall for {}: {}".format(label, sum/foldsCount))