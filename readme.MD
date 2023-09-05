# K-Means Text Clustering and Naive Bayes Classifier for Movie Reviews

This project implements K-means clustering to separate movie reviews based on the similarity of their word frequency distributions. Both Euclidean distance and Cosine similarity measures are utilized for clustering. Additionally, a Naive Bayes classifier is implemented using NLTK to categorize movie reviews as positive or negative.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Features](#features)
- [What I Learned](#what-i-learned)

## Getting Started
Clone the repository to your local machine to get started with the project.

## Prerequisites
The code relies on the following Python libraries, which need to be installed.

numpy
nltk
You can install them using pip:

bash
Copy code
pip install numpy nltk

## Usage
Make sure you have all the prerequisites installed.
Run the main Python script to see the K-means clustering and Naive Bayes classifier in action.
bash
Copy code
python main_script.py

## Features
K-means Clustering: The program uses K-means clustering to group movie reviews based on the similarity of their term frequency-inverse document frequency (TF-IDF) scores. It supports both Euclidean distance and Cosine similarity as distance measures.
Naive Bayes Classifier: A Naive Bayes Classifier is implemented to categorize movie reviews as positive or negative.
Natural Language Processing: The program uses NLTK for text processing and feature extraction. It incorporates Named Entity Recognition and Noun Phrase Chunking.
Data Preprocessing: Text data is cleaned and preprocessed using various techniques like tokenization, stemming, and removal of stopwords.

## What I Learned
Gained experience in implementing K-means clustering from scratch.
Learned about different distance metrics like Euclidean Distance and Cosine Similarity and their applications in text clustering.
Acquired hands-on experience with Natural Language Processing using the NLTK library, specifically in the areas of Named Entity Recognition and Noun Phrase Chunking.
Understood the workings of Naive Bayes classifiers and their applications in text categorization.
Learned data preprocessing techniques like tokenization, stemming, and stopwords removal for feature extraction in NLP tasks.