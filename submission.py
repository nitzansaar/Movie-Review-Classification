import math
from unittest import TestCase

from nltk import FreqDist
from nltk.corpus import movie_reviews

from kmeans import cosine_similarity, euclidean, tfidf, k_means, preprocess, get_corpus_freq


class Test(TestCase):

    def test_euclidean(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        print(euclidean(f1, f2))



    def test_cosine_similarity(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
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
        corpus_freqs = get_corpus_freq(sample_data)
        for word, freq in corpus_freqs.items():
            print(f"{word}: {freq}")
    def test_kmeans(self):
        positive_reviews = movie_reviews.fileids('pos')
        negative_reviews = movie_reviews.fileids('neg')
        list_of_files = positive_reviews[:10] + negative_reviews[:10]
        k = 2
        dist_measure = euclidean
        starting_method = 'random_seed'
        clusters = k_means(list_of_files, k, dist_measure, starting_method)
        print(clusters)
    def test_preprocess(self):
        positive_reviews = movie_reviews.fileids('pos')
        negative_reviews = movie_reviews.fileids('neg')
        list_of_files = positive_reviews[:10] + negative_reviews[:10]
        preprocessed_data = preprocess(list_of_files)
        print("Preprocessed data:")
        for file_data in preprocessed_data:
            print(f"{file_data[0]}: {file_data[1]}")







