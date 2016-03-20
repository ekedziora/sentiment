from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.corpus import movie_reviews
import nltk
import string
from nltk.corpus import stopwords
from nltk.metrics.association import BigramAssocMeasures
from nltk.metrics.association import TrigramAssocMeasures
from nltk.metrics.scores import precision, recall
from collections import defaultdict
from sklearn import cross_validation
from nltk import FreqDist, ConditionalFreqDist

stop = stopwords.words('english')

def normalizeWords(words):
    return [w.lower() for w in words if w.lower() not in stop and w.strip(string.punctuation)]

def findMostFrequentBigrams(words, scoreFunction=BigramAssocMeasures.chi_sq, count=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    return set(bigram_finder.nbest(scoreFunction, count))

def findMostFrequentTrigrams(words, scoreFunction=TrigramAssocMeasures.chi_sq, count=100):
    trigramFinder = TrigramCollocationFinder.from_words(words)
    return set(trigramFinder.nbest(scoreFunction, count))

def findBestWords(wordsInCategories, scoreFunction=BigramAssocMeasures.chi_sq, max_words=1000):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for category, words in wordsInCategories:
        word_fd.update(words)
        label_word_fd[category].update(words)

    word_counts = {}
    for condition in label_word_fd.conditions():
        word_counts[condition] = label_word_fd[condition].N()

    total_word_count = 0
    for condition, count in word_counts.items():
        total_word_count += count

    word_scores = {}

    for word, freq in word_fd.items():
        score = 0
        for condition, count in word_counts.items():
            score += scoreFunction(label_word_fd[condition][word], (freq, word_counts[condition]), total_word_count)
        word_scores[word] = score


    hapaxLegomenas = set([sample for sample in word_fd if word_fd[sample] == 1])
    best = sorted(word_scores.items(), key=lambda t: t[1], reverse=True)[:max_words]
    return set([w for w, s in best]) | hapaxLegomenas

def bagOfWordsFeatures(words, bestWords = set(), mostFrequentBigrams = set(), mostFrequentTrigrams = set()):
    words_normalized_list = normalizeWords(words)
    all_bigrams = set(nltk.bigrams(words_normalized_list))
    all_trigrams = set(nltk.trigrams(words_normalized_list))

    features = {}
    unigrams = (set(words_normalized_list) & set(bestWords)) if bestWords else set(words_normalized_list)
    bigrams = all_bigrams & mostFrequentBigrams if mostFrequentBigrams else all_bigrams
    trigrams = all_trigrams & mostFrequentTrigrams
    for unigram in unigrams:
        features[unigram] = True
    for bigram in bigrams:
        features[str(bigram)] = True
    for trigram in trigrams:
        features[str(trigram)] = True

    return features

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

def performCrossValidation(labels, foldsCount, debugMode):
    accurancySum = 0.0
    precisionSums = defaultdict(float)
    recallSums = defaultdict(float)
    crossValidationIterations = cross_validation.StratifiedKFold(labels, n_folds=foldsCount)
    for train, test in crossValidationIterations:
        trainset = [featureset[i] for i in train]
        testset = [featureset[i] for i in test]
        # classifier = nltk.NaiveBayesClassifier.train(trainset)
        # classifier = nltk.MaxentClassifier.train(trainset, algorithm='gis', trace=0, max_iter=20, min_lldelta=0.1)
        classifier = SklearnClassifier(NuSVC()).train(trainset)

        accurancy = nltk.classify.accuracy(classifier, testset)
        accurancySum += accurancy

        if debugMode:
            print("Accurancy: {}".format(accurancy))

        precisions, recalls = precision_recall(classifier, testset)

        for label, value in precisions.items():
            if debugMode:
                print("Precision for {}: {}".format(label, value))
            precisionSums[label] += value
        for label, value in recalls.items():
            if debugMode:
                print("Recall for {}: {}".format(label, value))
            recallSums[label] += value

    print("Average accurancy: {}".format(accurancySum/foldsCount))
    for label, sum in precisionSums.items():
        print("Average precision for {}: {}".format(label, sum/foldsCount))
    for label, sum in recallSums.items():
        print("Average recall for {}: {}".format(label, sum/foldsCount))

featureset = []
labels = []
foldsCount = 10

all_words_normalized = normalizeWords(movie_reviews.words())
wordsInCategories = [(label, normalizeWords(movie_reviews.words(categories=[label]))) for label in movie_reviews.categories()]
bestWords = findBestWords(wordsInCategories, max_words=500)
mostCommonBigrams = findMostFrequentBigrams(all_words_normalized, count=500)
# mostCommonTrigrams = findMostFrequentTrigrams(all_words_normalized, count=500)

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        features = bagOfWordsFeatures(movie_reviews.words(fileid), bestWords=bestWords, mostFrequentBigrams=mostCommonBigrams)
        featureset += [(features, category)]
        labels.append(category)

performCrossValidation(labels, foldsCount, False)