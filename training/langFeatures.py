
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize, word_tokenize
from enum import Enum
import string
import csv

# Was having issues with NLTK bc donwloads take a while. This clears that up
nltk.download('stopwords')

# 1 = English, 0 = Dutch


class Lang(Enum):
    english = 1
    dutch = 0


commonDutch = ['ik', 'je', 'dat', 'ze', 'hebben',
               'weet', 'kan', 'ja', 'nee', 'bent', 'doen']
dutchVowelCombo = ["uu", "aa", "ieu", "ij", "ooi", "oei"]
dutchSuf = ["ische", "thisch", "thie", "achtig",
            "aan", "iek", "ief", "ier", "iet", "een", "ant"]
dutchPre = []
dutch_possessive = ['a', 'i', 'o', 'u', 's']

commonEng = ['area', 'book', 'business', 'case', 'child', 'company', 'country', 'day', 'eye', 'fact', 'family', 'government', 'group',
             'hand', 'home', 'job', 'life', 'lot', 'man', 'money', 'month', 'mother', 'mr', 'night', 'number', 'part', 'people',
             'place', 'point', 'problem', 'program', 'question', 'right', 'room', 'school', 'state', 'story', 'student', 'study',
             'system', 'thing', 'time', 'water', 'way', 'week', 'woman', 'word', 'work', 'world', 'year']
engVowelCombo = ["aw", "ay", "oy", "kn", "ph"]
engSuf = ["tion", "sion", "ial", "able", "ible", "ful", "acy", "ance",
          "ism", "ity", "ness", "ship", "ish", "ive", "less", "ious", "ify"]
engPre = ["un"]


def containsDutchStop(sentence):
    """
    Checks if the sentence contains an dutch stop word.
    Return True if the sentence contains an dutch stop word
    and false otherwise
    """
    setStopwords = set(stopwords.words("dutch"))

    # Convert word into lower case and only consider
    # unique words
    setWords = set([word.lower() for word in sentence])

    commonEle = setWords.intersection(setStopwords)

    if len(commonEle) > 0:
        return 0

    return 1


def vowelComboDutch(sentence):
    """
    Check to see if any words in sentence contain
    the ij pairing
    """

    for word in sentence:
        for vowel in dutchVowelCombo:
            if vowel in word:
                return 0

    return 1


def wordEndDutch(sentence):
    """
    Check if any words in sentence ends with
    the ending provided by the dutch Lang

    :param sentence:
    :param end: The word that we want to end with
    :return:
    """

    for end in dutchSuf:
        endLen = len(end)

        for word in sentence:
            if len(word) >= endLen and word[-endLen:] == end:
                return 0

    return 1


def containsCommonDutch(sentence):
    """
    Determines whether the sentence
    contains that word. If it contains the word from the dutch list,
    return that it is a dutch word.
    :param sentence:
    :return:
    """

    for word in sentence:
        if word in commonDutch:
            return 0

    return 1


def containsEngStop(sentence):
    """
    Checks if the sentence contains an english stop word.
    Return True if the sentence contains an english stop word
    and false otherwise
    """
    setStopwords = set(stopwords.words("english"))

    # Convert word into lower case and only consider
    # unique words
    setWords = set([word.lower() for word in sentence])

    commonEle = setWords.intersection(setStopwords)

    if len(commonEle) > 0:
        return 1

    return 0


def vowelComboEng(sentence):

    for word in sentence:
        for vowel in engVowelCombo:
            if vowel in word:
                return 1

    # Nothing found, so return false
    return 0


def wordEndEng(sentence):
    """
    Check if any words in sentence ends with
    the ending provided by the english Lang
    """

    for end in engSuf:
        endLen = len(end)

        for word in sentence:
            if len(word) >= endLen and word[-endLen:] == end:
                return 1

    return 0


def containsCommonEng(sentence):
    """
    Determines whether the sentence
    contains that word. If it contains the word from the dutch list,
    return that it is a dutch word.
    :param sentence:
    :return:
    """

    for word in sentence:
        if word in commonEng:
            return 1

    return 0


def commonLetterEng(sentence):

    num_e = 0

    for word in sentence:
        if word in 'y':
            num_e += 1

    if num_e >= 10:
        return 1

    return 0


def containsPrefixEng(sentence):
    for word in sentence:
        if word[:-2] == 'un':
            return 1

    return 0


def possessiveEng(sentence):
    """
    Does it contain a possessive pronoun. If it does
    is the previous word a vowel or an s. If it is,
    it is dutch, otherwise it is English
    """
    # The sentence contains a possessive pronouns
    has_possessive = getIdx(sentence, "'s")

    if has_possessive != -1:
        return 1

    return 0


def cleanSentence(sentence):
    """
    Tokenize the sentence, and remove punctuation.
    Return a list of words in sentence.
    """
    sentence = sentence.translate(str.maketrans(
        '', '', string.punctuation))  # Strip punctuations

    tokens = wordpunct_tokenize(sentence)  # Tokenize sentence

    return tokens


def getIdx(sentence, word):
    for i, w in enumerate(sentence):
        if w == word:
            return sentence[i-1][-1], i

    return -1


def writeToCSV(des, src, test=False):
    if "test" in src:
        test = True

    title = ["CommonDutch", "CommonEng",
             "VowelComboDutch", "VowelComboEng",
             "StopDutch", "StopEng",
             "EndDutch", "EndEng",
             "Lang"]

    # Result for one sentence for each of the test cases
    resultRow = []

    # Write title
    # Write data to csv
    with open(des, "w") as data:
        writer = csv.writer(data)
        writer.writerow(title)

    # The training data
    raw = open(src, "r")

    for line in raw.readlines():

        # Only do this if it is not a test file
        # for training process
        if test is False:
            tokens = line.split("|")
            lang = tokens[0].strip()
            sent = tokens[1].strip()
        else:
            sent = line

        sentToken = cleanSentence(sent)
        resultRow.append(containsCommonDutch(sentToken))
        resultRow.append(containsCommonEng(sentToken))
        resultRow.append(vowelComboDutch(sentToken))
        resultRow.append(vowelComboEng(sentToken))
        resultRow.append(containsDutchStop(sentToken))
        resultRow.append(containsEngStop(sentToken))
        resultRow.append(wordEndDutch(sentToken))
        resultRow.append(wordEndEng(sentToken))

        if test is False:
            if lang == "en":
                resultRow.append(1)
            else:
                resultRow.append(0)

        # Write data to csv
        with open(des, "a") as data:
            writer = csv.writer(data)
            writer.writerow(resultRow)

        # Result row
        resultRow = []
