## Let's implement k-means and see how it does at separating the movie
## reviews. We'll try both euclidean distance and cosine similarity as
## our measures.
import random
from collections import defaultdict

import numpy as np
import math

from filters import *


## A cluster is a centroid, plus all the documents in that cluster.
class Cluster:
    def __init__(self):
        self.centroid = None
        self.members = []


## assume we are passing in two FreqDists
## d = sqrt(sum(a - b)^2)
## you implement this.
def euclidean(f1, f2):
    total = 0
    for word in f1:
        if word in f2:
            total += (f1[word] - f2[word]) ** 2
        else:
            total += (f1[word] ** 2)
    for word in f2:
        if word not in f1:
            total += (f2[word] ** 2)
    euc_dist = math.sqrt(total)
    return euc_dist


## assume we are passing in two FreqDists. You implement this.
## cos(f1,f2) = (f1 . f2) / (||f1|| ||f2||)
# how small the angle between the two vectors is. identical documents have theta = 0 & cos(0) = 1
def cosine_similarity(f1, f2):
    numerator = 0
    sum1 = 0
    for word in f1:
        sum1 += f1[word] ** 2
        if word in f2:
            numerator += f1[word] * f2[word]
    sum2 = 0
    for word in f2:
        sum2 += f2[word] ** 2
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    cos_sim = numerator / denominator
    return cos_sim


## assume that doc is a FreqDist representing a document, and corpus another FreqDist representing
## the fraction of documents that contain each word in our lexicon.
## return a new FreqDist that maps each word in doc onto its TFIDF score
def tfidf(doc, corpus):
    result = defaultdict(float)
    corpus_val = sum(corpus.values())
    for word, count in doc.items():
        tf = count / len(doc)
        df = corpus[word]
        idf = math.log(corpus_val / df)
        result[word] = tf * idf
    return result


## take in a list of filenames and return a list of tuples of the form
## [('name','FreqDist), ('name',FreqDist) ...]
def preprocess(filenames):
    results = []
    for name in filenames:
        word_list = apply_transforms(transforms, select_features(filters, movie_reviews.words(name)))
        model = FreqDist()
        for word in word_list:
            model[word] += 1
        results.append((name, model))
    return results


def get_corpus_freq(tuple_list):
    result = defaultdict(int)
    for filename, fd in tuple_list:
        for word in fd:
            result[word] += 1
    return result


def k_means(list_of_files, k, dist_measure, starting_method):
    ## preprocess files - get a list of (name, FreqDist tuples)
    tuple_list = preprocess(list_of_files)
    corpus_freqs = get_corpus_freq(tuple_list)
    ## compute TFIDF here. Get all the document frequencies.
    for i, doc in enumerate(tuple_list):
        doc_freq_dist = doc[1]
        doc_tfidf = tfidf(doc_freq_dist, corpus_freqs)
        tuple_list[i] = (doc[0], doc_tfidf)
    ## setup
    pos_cluster = Cluster()
    neg_cluster = Cluster()
    if starting_method == 'random_seed':
        seed1, seed2 = random.sample(tuple_list, 2)
        pos_cluster.centroid = seed1[1]
        neg_cluster.centroid = seed2[1]
        for doc in tuple_list:
            if doc == seed1 or doc == seed2:
                continue
            distance1 = dist_measure(doc[1], seed1[1])
            distance2 = dist_measure(doc[1], seed2[1])
            if distance1 < distance2:
                pos_cluster.members.append(doc)
            else:
                neg_cluster.members.append(doc)
    else:
        # random partition - assigns each document to a random cluster initially
        for doc in tuple_list:
            r = random.randint(0, 1)
            if r == 0:
                pos_cluster.members.append(doc)
            else:
                neg_cluster.members.append(doc)
    ## Our algorithm:
    ## while not converged:
    ##     compute the centroid of each cluster.
    ##     for each document, place it in the cluster with the
    ##     closest centroid.
    return tuple_list
