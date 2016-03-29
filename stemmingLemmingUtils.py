from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

lemmitizer = WordNetLemmatizer()

def translateFromNltkToWordnetTag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def doWordnetLemmatization(words):
    wordsTagged = nltk.pos_tag(words)
    lemmatized = []
    for word, tag in wordsTagged:
        wordnetTag = translateFromNltkToWordnetTag(tag)
        if wordnetTag:
            lemmatized.append(lemmitizer.lemmatize(word, wordnetTag))
        else:
            lemmatized.append(word)
    return lemmatized

def doStemming(words, stemmer):
    return [stemmer.stem(word) for word in words]