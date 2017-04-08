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
	  
data = pd.read_table("speech_data_extend.txt",encoding="utf-8")
raw_txt = str(data)
print(raw_txt)

################## Preprocess Data ####################

# 1.  tokenize data: split raw character string into individual elements of interest-- words, numbers, punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
print(word_tokenize(raw_txt))
word_tok = (word_tokenize(raw_txt))


# 2. Remove non-alphabetic characters


# 3. Remove stopwords using a list of your choice
from nltk.corpus import stopwords
set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in word_tok if not w in stop_words]
filtered_sentence = []

for w in word_tok:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tok)
print(filtered_sentence)


# 4. Stem the data using the Porter stemmer 

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# 5. Compute the corpus-level tf-idf score for every term, and choose a cutoﬀ below which to remove word 




# 6. Form the document-term matrix



################## Run Analysis ####################



################## Perform a SVD ####################



################## Program using EM Algorithm ####################