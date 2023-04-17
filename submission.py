import math
import random
from unittest import TestCase

import numpy as np
from nltk import FreqDist
from nltk.corpus import movie_reviews

from kmeans import cosine_similarity, euclidean, tfidf, k_means, preprocess, get_corpus_freq, Cluster


class Test(TestCase):

    def test_euclidean(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I': 1, 'loved': 1, 'Denzel': 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        print(euclidean(f1, f2))

    def test_cosine_similarity(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I': 1, 'loved': 1, 'Denzel': 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        print(cosine_similarity(f1, f2))

    def test_tfidf(self):
        doc = FreqDist(['apple', 'orange', 'orange', 'banana', 'apple', 'orange'])
        corpus = FreqDist({'apple': 2, 'orange': 3, 'banana': 1, 'mango': 2})
        tfidf_scores = tfidf(doc, corpus)
        for word, score in tfidf_scores.items():
            print(word, score)

    def test_compute_corpus_frequencies(self):
        sample_data = [
            ("file1", {"apple": 2, "banana": 3, "orange": 1}),
            ("file2", {"apple": 1, "banana": 4, "grape": 3}),
            ("file3", {"apple": 5, "orange": 2, "grape": 1})
        ]
        corpus_freq = get_corpus_freq(sample_data)
        for word, freq in corpus_freq.items():
            print(f"{word}: {freq}")

    def test_kmeans(self):
        positive_reviews = movie_reviews.fileids('pos')
        negative_reviews = movie_reviews.fileids('neg')
        list_of_files = positive_reviews[:50] + negative_reviews[:50]
        random.shuffle(list_of_files)
        # dist_measure = cosine_similarity
        dist_measure = euclidean
        k = 2
        starting_method = 'random_seed'
        # starting_method = 'random_partition'
        pos_percent_list = []
        neg_percent_list = []
        for i in range(5):
            pos_cluster, neg_cluster = k_means(list_of_files, k, dist_measure, starting_method)
            pos_count = 0
            for doc in pos_cluster:
                if doc[0] in positive_reviews:
                    pos_count += 1
            neg_count = 0
            for doc in neg_cluster:
                if doc[0] in negative_reviews:
                    neg_count += 1
            pos_percent = (pos_count / (len(list_of_files) / 2)) * 100
            neg_percent = (neg_count / (len(list_of_files) / 2)) * 100
            pos_percent_list.append(pos_percent)
            neg_percent_list.append(neg_percent)
        avg_pos_percent = np.mean(pos_percent_list)
        avg_neg_percent = np.mean(neg_percent_list)
        print(f"{starting_method}, {dist_measure.__name__}")
        # greater number to be used for accuracy
        if avg_pos_percent > avg_neg_percent:
            print(f"Average Positive reviews: {avg_pos_percent}%")
        else:
            print(f"Average Negative reviews: {avg_neg_percent}%")

    def test_preprocess(self):
        positive_reviews = movie_reviews.fileids('pos')
        negative_reviews = movie_reviews.fileids('neg')
        list_of_files = positive_reviews[:10] + negative_reviews[:10]
        preprocessed_data = preprocess(list_of_files)
        print("Preprocessed data:")
        for file_data in preprocessed_data:
            print(f"{file_data[0]}: {file_data[1]}")
