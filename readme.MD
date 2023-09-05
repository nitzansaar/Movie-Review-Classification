# Movie Review Classifier

This project implements K-means clustering to separate movie reviews based on the similarity of their word frequency distributions. Both Euclidean distance and Cosine similarity measures are utilized for clustering. Additionally, a Naive Bayes classifier is implemented using NLTK to categorize movie reviews as positive or negative.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Features](#features)
- [What I Learned](#what-i-learned)

## Prerequisites
The code relies on the following Python libraries, which need to be installed.

- numpy
- nltk

## Features
- K-means Clustering: The program uses K-means clustering to group movie reviews based on the similarity of their term frequency-inverse document frequency (TF-IDF) scores. It supports both Euclidean distance and Cosine similarity as distance measures.
- Naive Bayes Classifier: A Naive Bayes Classifier is implemented to categorize movie reviews as positive or negative.
- Natural Language Processing: The program uses NLTK for text processing and feature extraction. It incorporates Named Entity Recognition and Noun Phrase Chunking.
- Data Preprocessing: Text data is cleaned and preprocessed using various techniques like tokenization, stemming, and removal of stopwords.

## What I Learned
- Gained experience in implementing K-means clustering from scratch.
- Learned about different distance metrics like Euclidean Distance and Cosine Similarity and their applications in text clustering.
- Acquired hands-on experience with Natural Language Processing using the NLTK library, specifically in the areas of Named Entity Recognition and Noun Phrase Chunking.
- Understood the workings of Naive Bayes classifiers and their applications in text categorization.
- Learned data preprocessing techniques like tokenization, stemming, and stopwords removal for feature extraction in NLP tasks.