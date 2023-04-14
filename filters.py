from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from math import log

eng_words = stopwords.words("english")


def alphabetic(token):
    try:
        return token.isalpha()
    except:
        return False


def stopword(token):
    return token not in eng_words


# removes words that are less than 3 characters long
def short_or_long_word(token, min_length=3, max_length=8):
    return min_length <= len(token) <= max_length


filters = [alphabetic, stopword, short_or_long_word]


def replace_contractions(token):
    if "n't" in token:
        return "not"
    else:
        return token


def trim(token):
    try:
        return token.strip()
    except:
        return token


def lowercase(token):
    try:
        return token.lower()
    except:
        return token


def select_features(filters, list_of_tokens):
    features = []
    for token in list_of_tokens:
        if all([filter(token) for filter in filters]):
            features.append(token)
    return features


def remove_digits(token):
    result = ""
    for char in token:
        if not char.isdigit():
            result += char
    return result


def apply_transforms(transforms, list_of_tokens):
    changed = []
    for token in list_of_tokens:
        new_token = token
        for transform in transforms:
            new_token = transform(token)
        changed.append(new_token)
    return changed


transforms = [trim, lowercase, remove_digits, replace_contractions]
