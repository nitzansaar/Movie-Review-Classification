## Naive Bayes using NLTK
import random

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
def train(training_set):
    model = ConditionalFreqDist()
    for category in training_set:
        files = training_set[category]
        for name in files:
            word_list = apply_transforms(transforms, select_features(filters, movie_reviews.words(name)))
            for word in word_list:
                model[category][word] += 1
    return model


## classify
## Given a list of tokens and a model, return a dictionary mapping categories in the model
# to their log-likelihood.

def classify(model, list_of_tokens):
    filtered_tokens = apply_transforms(transforms, select_features(filters, list_of_tokens))
    categories = list(model.keys())
    results = {}
    ## you do the rest!
    for category in categories:
        log_likelihood = 0
        for token in filtered_tokens:
            count = model[category][token]
            log_likelihood += log((count + 1) / model[category].N() + len(filtered_tokens))
        results[category] = log_likelihood
    return results


## You will need to extend this to do five-fold cross-validation, and also compute accuracy.
if __name__ == "__main__":
    file_ids = movie_reviews.fileids()
    pos = [item for item in file_ids if item.startswith('pos')]
    neg = [item for item in file_ids if item.startswith('neg')]
    tset = {'pos': pos, 'neg': neg}
    random.shuffle(pos)
    random.shuffle(neg)
    accuracy_list = []
    for i in range(5):
        start_pos = int(i * len(pos) / 5)
        end_pos = int((i + 1) * len(pos) / 5)
        start_neg = int(i * len(neg) / 5)
        end_neg = int((i + 1) * len(neg) / 5)
        train_pos = pos[:start_pos] + pos[end_pos:]
        train_neg = neg[:start_neg] + neg[end_neg:]
        train_set = {'pos': train_pos, 'neg': train_neg}
        test_set = {'pos': pos[start_pos:end_pos], 'neg': neg[start_neg:end_neg]}
        model = train(train_set)
        correct = 0
        total = 0
        for category in test_set:
            for name in test_set[category]:
                result = classify(model, movie_reviews.words(name))
                true_val = category
                if result['pos'] > result['neg']:
                    predicted = 'pos'
                else:
                    predicted = 'neg'
                if predicted == true_val:
                    correct += 1
                total += 1
        accuracy = correct / total
        accuracy_list.append(accuracy)
        print("Fold %d Accuracy: %f" % (i + 1, accuracy))
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    print("Average Accuracy: %f" % avg_accuracy)
