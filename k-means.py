## Let's implement k-means and see how it does at separating the movie
## reviews. We'll try both euclidean distance and cosine similarity as
## our measures.

from filters import *

## A cluster is a centroid, plus all the documents in that cluster.
class Cluster :
    def __init__(self):
        self.centroid = None
        self.members = []


## assume we are passing in two FreqDists
## d = sqrt(sum(a - b)^2)
## you implement this.
def euclidean(f1, f2) :
    pass

## assume we are passing in two FreqDists. You implement this.
## cos(f1,f2) = (f1 . f2) / (||f1|| ||f2||)
def cosine_similarity(f1, f2) :
   pass

## assume that doc is a FreqDist representing a document, and corpus another FreqDist representing
## the fraction of documents that contain each word in our lexicon.
## return a new FreqDist that maps each word in doc onto its TFIDF score

def tfidf(doc, corpus) :
    pass


## take in a list of filenames and return a list of tuples of the form
## [('name','FreqDist), ('name',FreqDist) ...]
def preprocess(filenames) :
    results = []
    for name in filenames :
        word_list = apply_transforms(transforms,select_features(filters, movie_reviews.words(name)))
        model = FreqDist()
        for word in word_list :
            model[word] += 1
        results.append((name, model))
    return results


def k_means(list_of_files, k, dist_measure, starting_method) :

    ## preprocess files - get a list of (name, FreqDist tuples)
    tuple_list = preprocess(list_of_files)

    ## compute TFIDF here. Get all the document frequencies.

    ## setup
    pos_cluster = Cluster()
    neg_cluster = Cluster()

    if starting_method == 'random_seed' :
 ## select 2 files at random.
        pass
    else :
    ## random partition
        pass
    ## Our algorithm:
    ## while not converged:
    ##     compute the centroid of each cluster.
    ##     for each document, place it in the cluster with the
    ##     closest centroid.













