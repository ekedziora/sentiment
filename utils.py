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
from sklearn.naive_bayes import BernoulliNB
from featureExtractors import trigramsFeatures, bigramsFeatures, unigramsFeatures
import nltk

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


# returns featrues: unigrams (limited if constraint set is present), bigrams and trigrams if constraint sets are present
def bagOfWordsFeatures(wordsNormalized, bestWords = set(), mostFrequentBigrams = set(), mostFrequentTrigrams = set()):
    features = {}

    features.update(unigramsFeatures(wordsNormalized, bestWords=bestWords))

    if mostFrequentBigrams:
        features.update(bigramsFeatures(wordsNormalized, bestBigrams=mostFrequentBigrams))

    if mostFrequentTrigrams:
        features.update(trigramsFeatures(wordsNormalized, bestTrigrams=mostFrequentBigrams))

    return features

# returns unigrams with pos tags
def bestUnigramsWithPosFeatures(words, bestWords, normalizationFunction):
    wordsTagged = nltk.pos_tag(words)
    normalizedWords = normalizationFunction(wordsTagged)
    unigramsWithPos = [(word, tag) for word, tag in normalizedWords.items() if word in bestWords]
    return {str(unigramWithPos): True for unigramWithPos in unigramsWithPos}

# returns unigrams with pos tags from normalized words list
def bestUnigramsWithPosFeaturesTaggedForNormalizedWords(normalizedWords, bestWords):
    wordsTagged = nltk.pos_tag(normalizedWords)
    return {str((word, tag)): True for word, tag in wordsTagged if word in bestWords}

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

def createWordsInCategoriesDictionary(corpus, normalizationFunction):
    return [(label, normalizationFunction(corpus.words(categories=[label]))) for label in corpus.categories()]

def performCrossValidation(featureset, labels, foldsCount, debugMode):
    accuracySum = 0.0
    precisionSums = defaultdict(float)
    recallSums = defaultdict(float)
    crossValidationIterations = cross_validation.StratifiedKFold(labels, n_folds=foldsCount)
    for train, test in crossValidationIterations:
        trainset = [featureset[i] for i in train]
        testset = [featureset[i] for i in test]
        classifier = nltk.NaiveBayesClassifier.train(trainset)
        # classifier = nltk.MaxentClassifier.train(trainset, algorithm='gis', trace=0, max_iter=20, min_lldelta=0.1)
        # classifier = SklearnClassifier(NuSVC()).train(trainset)
        # classifier = SklearnClassifier(BernoulliNB()).train(trainset)

        classifier.show_most_informative_features(100)


        accuracy = nltk.classify.accuracy(classifier, testset)
        accuracySum += accuracy

        if debugMode:
            print("Accurancy: {}".format(accuracy))

        precisions, recalls = precision_recall(classifier, testset)

        for label, value in precisions.items():
            if debugMode:
                print("Precision for {}: {}".format(label, value))
            precisionSums[label] += value
        for label, value in recalls.items():
            if debugMode:
                print("Recall for {}: {}".format(label, value))
            recallSums[label] += value

    print("Average accurancy: {0:.3f}".format(accuracySum/foldsCount))
    precRecall = {label: (sum/foldsCount, recallSums.get(label)/foldsCount) for label, sum in precisionSums.items()}
    for label, (prec, recall) in precRecall.items():
        print("Average precision for {0}: {1:.3f}".format(label, prec))
        print("Average recall for {0}: {1:.3f}".format(label, recall))
        fmeasure = 2 * prec * recall/(prec + recall)
        print("Average f measure for {0}: {1:.3f}".format(label, fmeasure))