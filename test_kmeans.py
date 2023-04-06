import math
from unittest import TestCase

class Test(TestCase):

    def test_euclidean(self):
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        sum = 0
        for word in f1 :
            if word in f2 :
                print(f"{f1[word]} - {f2[word]}")
                sum += (f1[word] - f2[word]) ** 2
            else :
                print(f"{f1[word]} - 0")
                sum += (f1[word] ** 2)
        euc_dist = math.sqrt(sum)
        print(euc_dist)



    def test_cosine_similarity(self):
#         go through f1 and check if the word is in f2
#               if in both, multipy num in f1 * num in f2 then add that to the numerator
# go through each word in f1 and f2 get abs(sum of num of each word ** 2) then add both together
#                   that is the denominator
        f1 = {'it': 2, 'was': 1, 'a': 2, 'great': 2, 'movie': 1,
              'I' : 1, 'loved' : 1, 'Denzel' : 1, 'did': 1, 'job': 1}
        f2 = {'the': 2, 'movie': 1, 'was': 1, 'terrible': 1, 'but': 1,
              'effects': 1, 'were': 1, 'great': 1}
        numerator = 0
        sum1 = 0
        for word in f1 :
            sum1 += f1[word] ** 2
            if word in f2 :
                numerator += f1[word] * f2[word]
        # print(numerator)
        sum2 = 0
        for word in f2:
            sum2 += f2[word] ** 2
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        # print(denominator)
        cos_sim = numerator / denominator
        print(cos_sim)






