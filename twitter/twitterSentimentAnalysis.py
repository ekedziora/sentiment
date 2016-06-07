import csv
import os
import warnings

from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.tokenize.casual import TweetTokenizer
import pickle

from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile, RFE, RFECV, SelectFromModel
from sklearn.pipeline import Pipeline

from featureExtractors import *
from lexicons.mpqa.mpqaDictionary import MpqaDictionaryWrapper
from normalization import normalizeTwitterWordsWithExtraFeatures, normalizeTwitterWordsWithNegationHandle
from utils import findBestWords, findMostFrequentBigrams, findMostFrequentTrigrams, createWordsInCategoriesDictionary, performCrossValidation, performTestValidation

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC, LinearSVR, NuSVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron, LassoCV
from sklearn.tree import DecisionTreeClassifier

tweetTokenizer = TweetTokenizer(reduce_len=True, preserve_case=True, strip_handles=False)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/train', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
testcorpus = CategorizedPlaintextCorpusReader('corpus/standford/test', r'(pos|neg|neu)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
# corpus = CategorizedPlaintextCorpusReader('corpus/standford/sample', r'(pos|neg)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)
corpus = CategorizedPlaintextCorpusReader('corpus/3-way/datacopy', r'(\w+)-tweet[0-9]+\.txt', cat_pattern=r'(\w+)-tweet[0-9]+\.txt', word_tokenizer=tweetTokenizer)


def getfeatures(normalizedWords, extraNormalizedWords = list()):
    features = {}
    # wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    # features.update(trigramsFeatures(normalizedWords))
    # features.update(posTagsCountFeatures(wordsTagged))
    # features.update(sentiwordnetSentimentWordsCountFeatures(wordsTagged))
    # features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaObjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(extraTwitterFeaturesCount(extraNormalizedWords))
    return features

def getfeaturesTest(normalizedWords, extraNormalizedWords):
    features = {}
    wordsTagged = nltk.pos_tag(normalizedWords)
    features.update(unigramsFeatures(normalizedWords))
    features.update(bigramsFeatures(normalizedWords))
    features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
    # features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
    features.update(extraTwitterFeaturesCount(extraNormalizedWords))
    return features

def getFeaturesetFromPickle():
    with open('wordsTaggedToCategory-3way', 'rb') as pickFile:
        wordsTaggedWithCategory = pickle.load(pickFile)

    featureset = []
    for wordsTagged, category in wordsTaggedWithCategory:
        features = {}
        features.update(mpqaSentimentWordsCountFeatures(wordsTagged, mpqaDictionary))
        # features.update(mpqaSubjectivityWordsCountFeatures(wordsTagged, mpqaDictionary))
        featureset += [(features, category)]

    return featureset

featureset = []
labels = []
normalizationFunction = normalizeTwitterWordsWithNegationHandle

mpqaDictionary = MpqaDictionaryWrapper()

i = 1
for category in corpus.categories():
    for fileid in corpus.fileids(category):
        words = corpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        features = getfeatures(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        featureset += [(features, category)]
        labels.append(category)
        i += 1

print(i)


for i, featuresPlusCategory in enumerate(getFeaturesetFromPickle()):
    initFeatures, initCategory = featureset[i]
    features, category = featuresPlusCategory
    initFeatures.update(features)

sklearn_classifier = LinearSVC()
clf = LogisticRegression()
rfe = RFE(estimator=clf, step=400, verbose=5)
rfecv = RFECV(clf, step=0.1, scoring='accuracy', verbose=4)
sfm = SelectFromModel(LassoCV())
sp = SelectPercentile(score_func=chi2, percentile=60)
pipeline = Pipeline([('classifier', sklearn_classifier)])
uniqLabels = ['pos', 'neg', 'neu']
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    performCrossValidation(featureset, labels, 10, pipeline, uniqLabels)

testfeatureset = []

classifier = SklearnClassifier(pipeline).train(featureset)

features_number = len(classifier._vectorizer.vocabulary_)
print("Features number: " + str(features_number))

# with open("featuresdata/3way/uni-bi-extra-mpqa-subj.csv", "w", encoding='utf8') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(('feature name', 'score'))
#     featuresScores = {featurename: pipeline.steps[0][1].scores_[index] for featurename, index in classifier._vectorizer.vocabulary_.items()}
#
#     specialFeatures = ['extra_hashtag', 'extra_user_handle', 'extra_url', 'extra_retweet', "extra_pos_emoticon",
#                        "extra_neg_emoticon", "extra_emoticon", "exclamation_mark", "question_mark", "upper_case", "capitalized_word",
#                        'mpqa_weaksubj', 'mpqa_strongsubj', 'mpqa_negative', 'mpqa_positive', 'mpqa_neutral']
#
#     for name in specialFeatures:
#         csvwriter.writerow((name, featuresScores.get(name)))
#
#     withoutSpecial = {featurename: score for featurename, score in featuresScores.items() if featurename not in specialFeatures}
#     sortedScores = [name for name, score in sorted(withoutSpecial.items(), key=lambda x: x[1], reverse=True)][:100]
#     selectedScores = [(featurename, featuresScores[featurename]) for featurename in sortedScores]
#     for featurename, score in selectedScores:
#         csvwriter.writerow((featurename, score))

mid = '2step/sentiment'
newFolder = r'dumps/' + mid + '/logreg/60pct'
# newFolder = r'dumps/' + mid + '/logreg'
# newFolder = r'dumps/' + mid + '/linsvc'
# newFolder = r'dumps/' + mid + '/dectree'

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

# with open(newFolder + "/uni", 'wb+') as fileout:
# with open(newFolder + "/uni-bi", 'wb+') as fileout:
# with open(newFolder + "/uni-bi-extra", 'wb+') as fileout:
# with open(newFolder + "/uni-bi-extra-mpqa-senti", 'wb+') as fileout:
# with open(newFolder + "/uni-bi-extra-mpqa-subj", 'wb+') as fileout:
#     pickle.dump(classifier, fileout)

for category in testcorpus.categories():
    for fileid in testcorpus.fileids(category):
        words = testcorpus.words(fileids=[fileid])
        normalizedWords = normalizationFunction(words)
        extraNormalizedWords = normalizeTwitterWordsWithExtraFeatures(words)
        testfeatures = getfeaturesTest(normalizedWords, extraNormalizedWords=extraNormalizedWords)
        testfeatureset += [(testfeatures, category)]

# performTestValidation(featureset, testfeatureset, pipeline, uniqLabels)