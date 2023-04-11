import math
from unittest import TestCase

from nltk import FreqDist

from kmeans import cosine_similarity, euclidean, tfidf


class Test(TestCase):

    def test_euclidean(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        assert euclidean(f1, f2) == 4.69041575982343



    def test_cosine_similarity(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        assert cosine_similarity(f1, f2) == 0.27668578554642986

    def test_tfidf(self):
        doc = FreqDist(['apple', 'orange', 'orange', 'banana', 'apple', 'orange'])
        corpus = FreqDist({'apple': 2, 'orange': 3, 'banana': 1, 'mango': 2})
        tfidf_scores = tfidf(doc, corpus)
        for word, score in tfidf_scores.items():
            print(word, score)







