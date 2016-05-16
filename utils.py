from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk import FreqDist
from nltk import ConditionalFreqDist
from collections import defaultdict
from nltk.metrics.scores import precision, recall
from sklearn import cross_validation
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC, LinearSVR, NuSVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from featureExtractors import trigramsFeatures, bigramsFeatures, unigramsFeatures
import nltk, pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

    best = sorted(word_scores.items(), key=lambda t: t[1], reverse=True)[:max_words]
    return set([w for w, s in best])

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

def precision_recall_2step(polarClassifier, sentiClassifier, testFeatures, labels):
    trues = []
    predicted = []
    for feats, label in testFeatures:
        dec = polarClassifier.classify(feats)
        if dec == 'polar':
            observed = sentiClassifier.classify(feats)
        else:
            observed = 'neu'
        trues.append(label)
        predicted.append(observed)

    precisions, recalls, fscores, supports = precision_recall_fscore_support(trues, predicted, pos_label=None, labels=labels)
    accuracy = accuracy_score(trues, predicted)

    return precisions, recalls, fscores, accuracy

def precision_recall_2way_with_threshold(classifier, testFeatures, threshold):
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    probs = classifier.prob_classify_many([feats for (feats, label) in testFeatures])

    trues = 0
    for i, (feats, label) in enumerate(testFeatures):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        prob = probs[i]
        if prob.prob(observed) < threshold:
            observed = 'neu'
        testsets[observed].add(i)
        if observed == label:
            trues += 1

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    accuracy = float(trues)/len(testFeatures)

    return precisions, recalls, accuracy


def createWordsInCategoriesDictionary(corpus, normalizationFunction):
    return [(label, normalizationFunction(corpus.words(categories=[label]))) for label in corpus.categories()]

def performCrossValidation(featureset, labels, foldsCount, sklearnclassifier, uniqLabels):
    accuracySum = 0.0
    precisionSums = defaultdict(float)
    recallSums = defaultdict(float)
    fscoreSums = defaultdict(float)
    crossValidationIterations = cross_validation.StratifiedKFold(labels, n_folds=foldsCount)
    for train, test in crossValidationIterations:
        trainset = [featureset[i] for i in train]
        testset = [featureset[i] for i in test]
        classifier = SklearnClassifier(sklearnclassifier).train(trainset)

        true = [label for features, label in testset]
        predicted = classifier.classify_many([features for features, label in testset])

        precisions, recalls, fscores, support = precision_recall_fscore_support(true, predicted, pos_label=None, labels=uniqLabels)
        accuracy = accuracy_score(true, predicted)
        accuracySum += accuracy

        for label, value in zip(uniqLabels, precisions):
            precisionSums[label] += value
        for label, value in zip(uniqLabels, recalls):
            recallSums[label] += value
        for label, value in zip(uniqLabels, fscores):
            fscoreSums[label] += value

    print("Average accurancy: {0:.3f}".format(accuracySum/foldsCount))
    measures = {label: (sum/foldsCount, recallSums.get(label)/foldsCount, fscoreSums.get(label)/foldsCount) for label, sum in precisionSums.items()}
    for label, (prec, recall, fscore) in measures.items():
        print("Average precision for {0}: {1:.3f}".format(label, prec))
        print("Average recall for {0}: {1:.3f}".format(label, recall))
        print("Average f score for {0}: {1:.3f}".format(label, fscore))

def performTestValidation(trainset, testset, sklearnclassifier, uniqLabels):
        classifier = SklearnClassifier(sklearnclassifier).train(trainset)
        true = [label for features, label in testset]
        predicted = classifier.classify_many([features for features, label in testset])

        precisions, recalls, fscores, support = precision_recall_fscore_support(true, predicted, pos_label=None, labels=uniqLabels)
        accuracy = accuracy_score(true, predicted)

        print("Test accuracy: {0:.3f}".format(accuracy))
        measures = {label: (precision, recall, fscore) for label, precision, recall, fscore in zip(uniqLabels, precisions, recalls, fscores)}
        for label, (prec, recall, fscore) in measures.items():
            print("Precision for {0}: {1:.3f}".format(label, prec))
            print("Recall for {0}: {1:.3f}".format(label, recall))
            print("F score for {0}: {1:.3f}".format(label, fscore))

# def performTestValidationWithThreshold(trainset, testset, sklearnclassifier, threshold):
#     classifier = SklearnClassifier(sklearnclassifier).train(trainset)
#     precisions, recalls, accuracy = precision_recall_2way_with_threshold(classifier, testset, threshold)
#
#     print("Test accuracy: {0:.3f}".format(accuracy))
#     precRecall = {label: (precision, recalls.get(label)) for label, precision in precisions.items()}
#     for label, (prec, recall) in precRecall.items():
#         print("Precision for {0}: {1:.3f}".format(label, prec))
#         print("Recall for {0}: {1:.3f}".format(label, recall))
#         fmeasure = 2 * prec * recall/(prec + recall)
#         print("F measure for {0}: {1:.3f}".format(label, fmeasure))
