import nltk

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

def posTagsCountFeatures(normalizedWords):
    wordsTagged = nltk.pos_tag(normalizedWords)
    posTags = [tag for word, tag in wordsTagged]
    return {tag: posTags.count(tag) for tag in posTags}

def posTagsPrresenceFeatrues(normalizedWords):
    wordsTagged = nltk.pos_tag(normalizedWords)
    posTags = set([tag for word, tag in wordsTagged])
    return {tag: True for tag in posTags}