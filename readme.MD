### Assignment 4: Learning and Natural Language

#### Due April 14, 11:59pm.

To submit: Please check all code into your GitHub repo, along with a PDF containing the answers to the written questions. 
Also, please provide a submission.py that demonstrates all of your code.

For the coding portions, I've provided some structure; you should feel free to expand or improve upon it as you'd like. If you want to bring in pandas or numpy, or use the cross-validation module from sklearn, go for it! 


I strongly encourage you (as always!) to use test-driven programming. As you implement each function, 
think about what it should do, what the outputs should be for a few different inputs, and write yourself a test for it.


**Question 1. Probability.** 

Please answer these questions in a separate PDF. The following questions all have written components; please add your answers to those questions in the same file.

Recall our table of cat data, showing the distribution of cat fur colors and genders.


| Fur Color | Male  | Female |
  ----------|-------| -------
| black | 0.1   | 0.14 |
| grey | 0.12  | 0.02 | 
| tabby | 0.06  | 0.16 |
| white | 0.02  | 0.08 |
| calico | 0.12  | 0.18 |

1. *(5 points)* To begin, let's use this to compute some marginal probabilities.

What is the probability that a cat is:
- White or male?
- Grey or female?
- Female?
- calico and male, or tabby and female?
- White and grey?

2. *(5 points)* Now let's use this data to do some inference.

We know that 15% of male cats like to be brushed. P(likes_brushed | male) = 0.15

We know that 30% of the female cats like to be brushed. P(likes_brushed | female) = 0.3

We just met a new cat that likes to be brushed. What is the probability that it is female?
P(female | likes_brushed)?

**Question 2. Document Clustering with k-means.**

For this question, let's assume that we have been hired by Gleeson Library to build a tool that can sort documents into clusters based on similarity.
To begin, they've asked us to come up with a prototype. We've decided to implement this prototype using the k-means algorithm. 
We'll test out our algorithm on NLTK's movie_reviews dataset. 

I've provided you with a setup to get you started. 

Clustering has a number of different design choices that we need to consider. The purpose of this question is to test these out and determine which choices 
are most effective for our problem. 

1) *(10 points)* The first question to consider is how to measure distance. We've learned about two approaches: Euclidean distance and cosine similarity. I've provided stubs for these functions. Implement both of them.

2) *(10 points)* Recall that vector-based measures tend to focus on common words. One way to address this is through TF-IDF weighting.
I've provided the stub of a function that takes as input two FreqDists: one representing the count of words in a document and one representing the document frequencies. (the fraction of documents in the corpus containing a word)
Implement this function and add it to preprocess. 

3) *(10 points)* The next question to consider is how to seed the initial clusters. Recall that there are two approaches: Random seed and Random Partition. 
- Random seed will choose two documents randomly to be the initial seeds, and then place every other document in cluster with the closest seed.
- Random partition assigns each document to a random cluster initially. 

Implement both of these as an if/else within k-means.

4) *(10 points)* Next, we need to think about feature selection. This is a very important problem when we're dealing with text; if we can identify and properly weight the important features, we can greatly improve performance.
I've provided some initial filters and transformations in filters.py, along with some code to allow you to dynamically reconfigure them.

*Filters* will return a token if it matches their filter, and False otherwise. For example, alphabetic filters words that contain only alphabetic characters,
and stopword returns a token if it is not a stopword.

*Transforms* will take a token as input and return a new, transformed token. For example, trim() removes whitespace, and lowercase converts to lower case.

Look at the movie_reviews data - what do you see in there that might be helpful? Use this to create at least three filters or transforms that you think will improve the performance of k-means.

5) *(10 points)* Now you're ready to finish implementing k_means. I've provided a starting point for you. Once you have it running, experiment with your filters, transforms, distance measures, and starting conditions to find the configuration that maximizes performance. Add a paragraph to your PDF file describing the choices that lead to the best performance.
(note: to measure performance: once the algorithm has converged, count the fraction of articles in each cluster that are of the same class. So if cluster 1 has 42 positive articles and 8 negative articles,
and cluster 2 has 38 negative articles and 12 positive articles, the accuracy is (42/50 + 38/50) / 2 = 40/50.)


Part 3: Document classification with Naive Bayes

Now let's change the problem a little, and see how we can do if we take advantage of the labels on the data.

1. *(10 points)* nb.py contains most of the code for building a Naive Bayes classifier. You just need to complete classify, using log-likelihood to return a dictionary mapping categories (pos, neg) to their likelihoods.
You will probably also want to add some functionality to the main.

2. *(10 points)* As with clustering, a big part of getting Naive Bayes to work well is selecting and transforming our features.
Implement at least four filters or transforms (you may re-use your ideas from k-means if you'd like) and test the performance of your classifier using five-fold cross-validation. (you can re-use your code from assignment 3, 
use sklearn, or re-implement this, as you prefer.)
*note: do NOT use TF-IDF for this part. It will give you all kinds of weird results.)

In your writeup, add a paragraph that describes the most effective set of filters and transforms, and the accuracy you were able to achieve using these.

**Question 4**. Named Entity Extraction. An alternative approach to textual analysis is to explicitly look at the structure of the sentences using a parser.
NLTK has a set of tools included that allow us to do this. The file chunking.py shows how it works with a simple example.

1) *(10 points)* To begin, find at least five English-language sentences from news sources to use as input. Ones that talk about specific people, companies or places are best.
Run them through the RegexpParser shown in chunking.py. This will extract _noun phrases_. Add a paragraph to your writeup indicating which phrases were correctly 
extracted, and which were not. Do you see any pattern to the phrases that were missed?

2) *(10 points)* Next, run the named entity chunker over your same examples. This will extract _named entities_, which 
are noun phrases that refer to specific people, places or things. Add a paragraph to your writeup indicating which
noun phrases were correctly extracted. Do you see a pattern to the phrases that were missed?

**(686 Students only)** Please read [this article](https://thegradient.pub/machine-learning-wont-solve-the-natural-language-understanding-challenge/) and answer the following questions:
 
- According to the author, what is the difference between natural language processing and natural language understanding?

- Why does the author feel that data-driven approaches are not suitable for NLU?

- If we accept the author's premise, what sorts of tasks are then best suited for data-driven approaches to NLP?

 

