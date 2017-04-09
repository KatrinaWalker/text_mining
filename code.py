#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 07:51:31 2017

@author: k2
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

nltk.download() # A box will pop up - download all

# with open("speech_data_extend.txt", 'r') as myfile:
    #data=myfile.read().replace('\n', '')

data = pd.read_table("speech_data_extend.txt", encoding="utf-8") # dataframe with 3 columns rows are sentences
len(data)
data = data.to_string()

################## Preprocess Data ####################

# 1.  tokenize data: split raw character string into individual elements of interest-- words, numbers, punctuation
from nltk.tokenize import word_tokenize
word_tok = (word_tokenize(data))
len(word_tok)
type(word_tok)
word_tok = str(word_tok)

# 2. Remove non-alphabetic characters
alpha = str.lower(word_tok)
len(alpha)

def stripNonAlphaNum(alpha):
    import re
    return re.compile(r'\W+', re.UNICODE).split(alpha)

wordlist = stripNonAlphaNum(alpha)
len(wordlist)

# 3. Remove stopwords using a list of your choice
from nltk.corpus import stopwords
set(stopwords.words('english'))

stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in wordlist if not w in stop_words]
filtered_sentence = []

for w in wordlist:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)
len(filtered_sentence)


# 4. Stem the data using the Porter stemmer 
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
list = []

for w in filtered_sentence:
    stem = list.append(ps.stem(w))
len(list)

stem = str(list)

# 5. Compute the corpus-level tf-idf score for every term, and choose a cutoﬀ below which to remove word 
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(list)
print("Vocabulary:", count_vectorizer.vocabulary_)


# 6. Form the document-term matrix

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)
tf_idf_matrix = tfidf.transform(term_freq_matrix)
print(tf_idf_matrix.todense())

################## Run Analysis ####################



################## Perform a SVD ####################



################## Program using EM Algorithm ####################