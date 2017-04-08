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
data.columns
raw_txt = data
raw_txtcolumns

################## Preprocess Data ####################

# 1.  tokenize data: split raw character string into individual elements of interest-- words, numbers, punctuation
from nltk.tokenize import sent_tokenize, word_tokenize

word = word_tokenize(raw_txt)
word = print(sent_tokenize(raw_txt))

# 2. Remove non-alphabetic characters


# 3. Remove stopwords using a list of your choice



# 4. Stem the data using the Porter stemmer 




# 5. Compute the corpus-level tf-idf score for every term, and choose a cutoﬀ below which to remove word 




# 6. Form the document-term matrix



################## Run Analysis ####################



################## Perform a SVD ####################



################## Program using EM Algorithm ####################