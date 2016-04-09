import nltk
import collections
from normalization import hashtag_code, url_code, user_handle_code
from stemmingLemmingUtils import translateFromNltkToWordnetTag
from nltk.corpus import sentiwordnet

def unigramsFeatures(normalizedWords, bestWords = set()):
    if bestWords:
        unigrams = set(normalizedWords) & set(bestWords)
    else:
        unigrams = set(normalizedWords)

    return {unigram: True for unigram in unigrams}

def bigramsFeatures(normalizedWords, bestBigrams = set()):
    allBigrams = set(nltk.bigrams(normalizedWords))

    if bestBigrams:
        bigrams = allBigrams & bestBigrams
    else:
        bigrams = allBigrams

    return {str(bigram): True for bigram in bigrams}

def trigramsFeatures(normalizedWords, bestTrigrams = set()):
    allTrigrams = set(nltk.trigrams(normalizedWords))

    if bestTrigrams:
        trigrams = allTrigrams & bestTrigrams
    else:
        trigrams = allTrigrams

    return {str(trigram): True for trigram in trigrams}

def sentiwordnetSentimentScoreFeatures(wordsTagged):
    posScoreSum = 0.0
    negScoreSum = 0.0
    wordsCount = 0
    for word, tag in wordsTagged:
        wordnetTag = translateFromNltkToWordnetTag(tag)
        if wordnetTag:
            synsets = list(sentiwordnet.senti_synsets(word, wordnetTag))
        else:
            synsets = list(sentiwordnet.senti_synsets(word))
        if len(synsets) > 0:
            synset = synsets[0]
            wordsCount += 1
            posScoreSum = synset.pos_score()
            negScoreSum = synset.neg_score()

    posScore = 0 if wordsCount == 0 else posScoreSum/wordsCount
    negScore = 0 if wordsCount == 0 else negScoreSum/wordsCount
    return {"pos_score": posScore, "neg_score": negScore}

def sentiwordetSentimentWordsPresenceFeatures(wordsTagged):
    features = {}
    for word, tag in wordsTagged:
        wordnetTag = translateFromNltkToWordnetTag(tag)
        if wordnetTag:
            synsets = list(sentiwordnet.senti_synsets(word, wordnetTag))
            if not synsets:
                synsets = list(sentiwordnet.senti_synsets(word))
        else:
            synsets = list(sentiwordnet.senti_synsets(word))
        if len(synsets) > 0:
            synset = synsets[0]
            if synset.pos_score() > 0.5:
                features["pos_word_presence"] = True
            if synset.neg_score() > 0.5:
                features["neg_word_presence"] = True
    return features

def sentiwordnetSentimentWordsCountFeatures(wordsTagged):
    features = collections.defaultdict(int)
    for word, tag in wordsTagged:
        wordnetTag = translateFromNltkToWordnetTag(tag)
        if wordnetTag:
            synsets = list(sentiwordnet.senti_synsets(word, wordnetTag))
        else:
            synsets = list(sentiwordnet.senti_synsets(word))
        if len(synsets) > 0:
            synset = synsets[0]
            if synset.pos_score() > 0.5:
                features["pos_word_count"] = features["pos_word_presence"] + 1
            if synset.neg_score() > 0.5:
                features["neg_word_count"] = features["neg_word_presence"] + 1
    return features

def posTagsCountFeatures(wordsTagged):
    posTags = [tag for word, tag in wordsTagged]
    return {tag: posTags.count(tag) for tag in posTags}

def posTagsRatioFeatures(normalizedWords):
    wordsTagged = nltk.pos_tag(normalizedWords)
    posTags = [tag for word, tag in wordsTagged]
    return {tag: posTags.count(tag)/len(normalizedWords) for tag in posTags}

def posTagsPresenceFeatrues(wordsTagged):
    posTags = set([tag for word, tag in wordsTagged])
    return {tag: True for tag in posTags}

def extraTwitterFeaturesCount(normalizedWords):
    features = collections.defaultdict(int)
    for word in normalizedWords:
        if word == hashtag_code:
            features["hashtag"] = features["hashtag"] + 1
        elif word == user_handle_code:
            features["user_handle"] = features["user_handle"] + 1
        elif word == url_code:
            features["url"] = features["url"] + 1
        elif "!" in word:
            features["exclamation_mark"] = features["exclamation_mark"] + word.count("!")
        elif "?" in word:
            features["question_mark"] = features["question_mark"] + word.count("?")
        elif word.isupper():
            features["upper_case"] = features["upper_case"] + 1
        elif word.istitle():
            features["capitalized_word"] = features["capitalized_word"] + 1
    return features

def extraTwitterFeaturesPresence(normalizedWords):
    features = {}
    for word in normalizedWords:
        if word == hashtag_code:
            features["extra_hashtag"] = True
        elif word == user_handle_code:
            features["extra_user_handle"] = True
        elif word == url_code:
            features["extra_url"] = True
        elif "!" in word:
            features["extra_exclamation_mark"] = True
        elif "?" in word:
            features["extra_question_mark"] = True
        elif word.isupper():
            features["extra_upper_case"] = True
        elif word.istitle():
            features["extra_capitalized_word"] = True
    return features