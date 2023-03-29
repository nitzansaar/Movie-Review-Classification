## Naive Bayes using NLTK
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from math import log
from filters import *


## assume the training set is:
## a dictionary mapping category names to lists of files.
## e.g. t = {'pos' : ['pos/cv999_13106.txt', 'pos/cv998_14111.txt',...] ,
##           'neg' : ['neg/cv371_8197.txt', 'neg/cv374_26455.txt'] }
## Construct a ConditionalFrequencyDistribution that maps categories onto distributions.

## This should just work. You're welcome to adapt it if you'd like.
def train(training_set) :

    model = ConditionalFreqDist()
    for category in training_set :
        files = training_set[category]
        for name in files :
            word_list = apply_transforms(transforms,select_features(filters, movie_reviews.words(name)))
            for word in word_list :
                model[category][word] += 1
    return model


## classify
## Given a list of tokens and a model, return a dictionary mapping categories in the model
# to their log-likelihood.

def classify(model, list_of_tokens) :
    filtered_tokens = apply_transforms(transforms, select_features(filters, list_of_tokens))
    categories = list(model.keys())
    results = {}
    ## you do the rest!


    return results

## You will need to extend this to do five-fold cross-validation, and also compute accuracy.
if __name__ == "__main__" :
    fileids = movie_reviews.fileids()
    pos = [item for item in fileids if item.startswith('pos')]
    neg = [item for item in fileids if item.startswith('neg')]
    tset = {'pos' : pos, 'neg' : neg}
    model = train(tset)

    for fileid in fileids :
        result = classify(model, movie_reviews.words(fileid))
        true_val = fileid[0:3]
        if result['pos'] > result['neg'] :
            predicted = 'pos'
        else :
            predicted = 'neg'
        print("%s %s" % (predicted, true_val))

