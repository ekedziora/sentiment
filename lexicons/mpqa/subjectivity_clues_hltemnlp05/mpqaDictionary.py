dictionary = {}
anyPosTag = "anypos"

def initDictionary():
    file = open("C:\\Users\\ekedz\\PycharmProjects\\sentiment\\lexicons\\mpqa\\subjectivity_clues_hltemnlp05\\subjclueslen1-HLTEMNLP05.tff", encoding='utf8')

    for line in file.read().splitlines():
        pairs = line.split()
        entry = {}
        for pair in pairs:
            elements = pair.split("=")
            entry[elements[0]] = elements[1]
        dictionary[(entry["word1"], entry["pos1"])] = (entry["type"], entry["priorpolarity"]) # todo stemmed

    file.close()

def getPolarity(word, posTag):
    polarity = None
    if posTag:
        polarity = dictionary.get((word, posTag))
    if polarity is None:
        polarity = dictionary.get((word, anyPosTag))
    return polarity[1] if polarity is not None else None

def getObjectivity(word, posTag):
    polarity = None
    if posTag:
        polarity = dictionary.get((word, posTag))
    if polarity is None:
        polarity = dictionary.get((word, anyPosTag))

    if polarity is None:
        return None
    if polarity[1] == 'positive' or polarity[1] == 'negative' or polarity[1] == 'both':
        return 'subjective'
    else:
        return 'objective'

def getSubjectivity(word, posTag):
    value = None
    if posTag:
        value = dictionary.get((word, posTag))
    if value is None:
        value = dictionary.get((word, anyPosTag))

    return value[0] if value is not None else None