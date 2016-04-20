import nltk
import collections
from normalization import hashtag_code, url_code, user_handle_code, retweet_code, positive_emoticon_code, negative_emoticon_code, emoticon_code, isWordNegated
from stemmingLemmingUtils import translateFromNltkToWordnetTag, translateFromNltkToMpqaTag
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

def bestUnigramsWithPosFeaturesTaggedForNormalizedWords(normalizedWords, bestWords):
    wordsTagged = nltk.pos_tag(normalizedWords)
    return {str((word, tag)): True for word, tag in wordsTagged if word in bestWords}

def sentiwordnetSentimentScoreFeatures(wordsTagged):
    posScoreSum = 0.0
    negScoreSum = 0.0
    for word, tag in wordsTagged:
        wordnetTag = translateFromNltkToWordnetTag(tag)
        if wordnetTag:
            synsets = list(sentiwordnet.senti_synsets(word, wordnetTag))
        else:
            synsets = list(sentiwordnet.senti_synsets(word))
        if len(synsets) > 0:
            synset = synsets[0]
            posScoreSum = synset.pos_score()
            negScoreSum = synset.neg_score()

    return {"pos_neg_score": posScoreSum - negScoreSum}

def sentiwordnetSentimentWordsPresenceFeatures(wordsTagged):
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

def mpqaSentimentWordsCountFeatures(wordsTagged, dictionary):
    features = collections.defaultdict(int)
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        polarity = dictionary.getPolarity(word, mpqaTag)
        if polarity is not None:
            features["mpqa_" + polarity] = features["mpqa_" + polarity] + 1
    return features

def mpqaSentimentWordsPresenceFeatures(wordsTagged, dictionary):
    features = {}
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        polarity = dictionary.getPolarity(word, mpqaTag)
        if polarity is not None:
            features["mpqa_" + polarity] = True
    return features

def mpqaObjectivityWordsScoreFeatures(wordsTagged, dictionary):
    features = collections.defaultdict(int)
    wordsCount = 0
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        objectivity = dictionary.getObjectivity(word, mpqaTag)
        if objectivity is not None:
            wordsCount += 1
            features["mpqa_" + objectivity] = features["mpqa_" + objectivity] + 1
    return {key: value/wordsCount if wordsCount > 0 else 0 for key, value in features.items()}

def mpqaObjectivityWordsPresenceFeatures(wordsTagged, dictionary):
    features = {}
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        objectivity = dictionary.getObjectivity(word, mpqaTag)
        if objectivity is not None:
            features["mpqa_" + objectivity] = True
    return features

def mpqaObjectivityWordsCountFeatures(wordsTagged, dictionary):
    features = collections.defaultdict(int)
    wordsCount = 0
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        objectivity = dictionary.getObjectivity(word, mpqaTag)
        if objectivity is not None:
            wordsCount += 1
            features["mpqa_" + objectivity] = features["mpqa_" + objectivity] + 1
    return features

def mpqaSubjectivityPresenceFeatures(wordsTagged, dictionary):
    features = {}
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        subjectivity = dictionary.getSubjectivity(word, mpqaTag)
        if subjectivity is not None:
            features["mpqa_" + subjectivity] = True
    return features

def mpqaSubjectivityWordsScoreFeatures(wordsTagged, dictionary):
    features = collections.defaultdict(int)
    wordsCount = 0
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        subjectivity = dictionary.getSubjectivity(word, mpqaTag)
        if subjectivity is not None:
            wordsCount += 1
            features["mpqa_" + subjectivity] = features["mpqa_" + subjectivity] + 1
    return {key: value/wordsCount if wordsCount > 0 else 0 for key, value in features.items()}

def mpqaSubjectivityWordsCountFeatures(wordsTagged, dictionary):
    features = collections.defaultdict(int)
    wordsCount = 0
    for word, tag in wordsTagged:
        mpqaTag = translateFromNltkToMpqaTag(tag)
        subjectivity = dictionary.getSubjectivity(word, mpqaTag)
        if subjectivity is not None:
            wordsCount += 1
            features["mpqa_" + subjectivity] = features["mpqa_" + subjectivity] + 1
    return features

def posTagsCountFeatures(wordsTagged):
    posTags = [tag for word, tag in wordsTagged]
    return {tag: posTags.count(tag) for tag in posTags}

def posTagsRatioFeatures(wordsTagged):
    posTags = [tag for word, tag in wordsTagged]
    return {tag: posTags.count(tag)/len(wordsTagged) for tag in posTags}

def posTagsPresenceFeatrues(wordsTagged):
    posTags = set([tag for word, tag in wordsTagged])
    return {tag: True for tag in posTags}

def extraTwitterFeaturesCount(normalizedWords):
    features = collections.defaultdict(int)
    for word in normalizedWords:
        if word == hashtag_code:
            features["extra_hashtag"] = features["extra_hashtag"] + 1
        elif word == user_handle_code:
            features["extra_user_handle"] = features["extra_user_handle"] + 1
        elif word == url_code:
            features["extra_url"] = features["extra_url"] + 1
        elif word == retweet_code:
            features["extra_retweet"] = features["extra_retweet"] + 1
        elif word == positive_emoticon_code:
            features["extra_pos_emoticon"] = features["extra_pos_emoticon"] + 1
        elif word == negative_emoticon_code:
            features["extra_neg_emoticon"] = features["extra_neg_emoticon"] + 1
        elif word == emoticon_code:
            features["extra_emoticon"] = features["extra_emoticon"] + 1
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
        elif word == retweet_code:
            features["extra_retweet"] = True
        elif word == positive_emoticon_code:
            features["extra_pos_emoticon"] = True
        elif word == negative_emoticon_code:
            features["extra_neg_emoticon"] = True
        elif word == emoticon_code:
            features["extra_emoticon"] = True
        elif "!" in word:
            features["extra_exclamation_mark"] = True
        elif "?" in word:
            features["extra_question_mark"] = True
        elif word.isupper():
            features["extra_upper_case"] = True
        elif word.istitle():
            features["extra_capitalized_word"] = True
    return features