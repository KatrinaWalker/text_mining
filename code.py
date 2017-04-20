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
import string



#nltk.download() # A box will pop up - download all- will take a little time

data_origin = pd.read_table("speech_data_extend.txt", encoding="utf-8") # dataframe with 3 columns -  rows are paragraphs
#len(data)
#data = data.to_string()
#data.shape

# Extract only the second column of the table
data = data_origin.iloc[:,1]
#print(data)
#data.info()

# Convert each document into lowercase
data = [element.lower() for element in data]

# Join all documents into a single text, separated by spaces
#data = " ".join(data_origin)
################## Preprocess Data ####################

# 1.  tokenize data: split raw character string into individual elements of interest-- words, numbers, punctuation
from nltk.tokenize import word_tokenize
data = [word_tokenize(element) for element in data]
len(data)
type(data)


# 2. Remove non-alphabetic characters

# Get rid of all empty entries, punctuation and numbers in the list
data = [[ w for w in doc if (w != '' and w not in string.punctuation and not any(char.isdigit() for char in w)) ] for doc in data]
len(data)

# 3. Remove stopwords using a list of your choice
from nltk.corpus import stopwords
#set(stopwords.words('english'))

stop_words = set(stopwords.words('english'))
data = [[ w for w in doc if w  not in stop_words ] for doc in data]


# 4. Stem the data using the Porter stemmer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
data = [[ ps.stem(w) for w in doc] for doc in data]


# 5 & 6 . Compute the corpus-level tf-idf score for every term, and choose a cutoﬀ below which to remove word - Form the document-term matrix

# Create a new list with all words, by collapsing the sublists in "data" into a single list
all_words = sum(data, [])

# Initialise dictionary to count document frequency of each term
df = {key: 0 for key in all_words}
#print(df)

# Iterator: for each word, iterate over every document. If word is present in text, +1 to value for that word.
for key,value in df.items():
    for doc in data:
        if key in doc:
            df[key] +=1 # add one for each document for which word appears


# Initialise a dictionary to compute the idf for each term
idf = {key: 0 for key in all_words}


ndocs = len(data)

# Compute idf for each word
for key,value in df.items():
    idf[key] = np.log(ndocs/df[key])

# Compute term frequency for each term
# Initialise dictionary
tf = {key: 0 for key in all_words}

#print(all_words)

from collections import Counter
tf = Counter(x for sublist in data for x in sublist)

for key,value in tf.items():
    tf[key] = 1 + np.log(tf[key])

# Calculate corpus-wide tf-idf for each term
tf_idf = {key: 0 for key in all_words}
for key,value in tf_idf.items():
    tf_idf[key] = tf[key]*idf[key]

# Plot corpus-wide tf_idf
y = sorted(tf_idf.values(), reverse = True)
plt.plot(y)
plt.show()

# Select only the terms with frequency higher than 20. Create a new dictionary and plot it.
d = {k: v for k, v in tf_idf.items() if v > 25}
len(d)

y = sorted(d.values(), reverse = True)
plt.plot(y)
plt.show()

# Save new words under new_words
new_words = [key for key in d]


# Construct a document-term matrix
dt_matrix = pd.DataFrame([[doc.count(w) for doc in data] for w in new_words])
dt_matrix.index = new_words
dt_matrix.shape

# Save the document-term matrix for future use
dt_matrix.to_csv("dt_matrix.csv", sep = ";", index=True)

# Retrieve the saved document-term matrix
dt_matrix = pd.DataFrame.from_csv("dt_matrix.csv", sep = ";", index_col = 0)


################## Run Analysis ####################


# A create dictionary to assess heterogeneaity
d = pd.read_excel("LoughranMcDonald_MasterDictionary_2014.xlsx")
#d.shape
#d.dtypes
dic = d[['Word', 'Negative']].copy()

dic = dic[dic.Negative != 0]
dic = dic[['Word']].copy()
dic_stemmer = PorterStemmer()

# test
#stemmer = PorterStemmer()
#plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
          # 'died', 'agreed', 'owned', 'humbled', 'sized',
          #  'meeting', 'stating', 'siezing', 'itemization',
           #'sensational', 'traditional', 'reference', 'colonizer',
          #'plotted']
#singles = [stemmer.stem(plural) for plural in plurals]
#print(' '.join(singles))

dic_list = dic['Word'].tolist()
dic_s = [str(w).lower() for w in dic_list]
dic = [dic_stemmer.stem(x) for x in dic_s]
dic = set(dic)
dic = [x for x in iter(dic)]
#dic = pd.DataFrame({'neg': dic})

data_2 = [' '.join(doc) for doc in data]

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=dic)
cv_results = cv.fit_transform(data_2)
cv_results_test = cv_results.todense()
cv_freq = np.sum(cv_results_test, axis = 1)
cv_freq_norm = cv_freq
for i in np.arange(len(data)):
    cv_freq_norm[i] = 100*(cv_freq_norm[i]/len(data[i]))

################## Perform a SVD ####################

# Generate a tf-idf-weighted document-term matrix

# Generate a term-frequency matrix
tf = pd.DataFrame(np.where(dt_matrix == 0, 0, 1 + np.log(dt_matrix)))
tf.index = dt_matrix.index

# Create a tf_idf matrix
tf_idf = tf
#tf_idf.index = tf.index

for w in tf_idf.index:
    tf_idf.loc[w] = tf.loc[w]* idf[w]

# Save the tf_idf matrix to a csv file for future use
tf_idf.to_csv("/Users/davidrosenfeld/Documents/text_mining_course/text_mining/tf_idf.csv", sep = ";", index=True)

tf_idf = pd.DataFrame.from_csv("/Users/davidrosenfeld/Documents/text_mining_course/text_mining/tf_idf.csv", sep = ";", index_col = 0)
tf_idf = np.transpose(tf_idf)


# Compute an SVD of the tf_idf matrix
U, s, V = np.linalg.svd(tf_idf)
U.shape, s.shape, V.shape

# Calculate and plot the proportion of variance explained by the singular values
pve = s/sum(s)
plt.plot(pve)
plt.show()

# Construct an approximate tf_idf matrix with the first 200 principal components
U_hat = U[:,0:200]
s_hat = s[0:200]
V_hat = V[0:200, 0:200]
tf_idf_hat = np.matmul(np.matmul(U_hat, np.diag(s_hat)), V_hat)
tf_idf_hat.shape

# Create a dataframe with years where most months were in recession as 1 and 0 otherwise
recession = pd.read_csv("/Users/davidrosenfeld/Documents/text_mining_course/text_mining/USREC.csv", index_col = False)
recession['DATE'] = pd.to_datetime(recession['DATE'], format='%Y-%m-%d')
recession = recession.groupby(recession['DATE'].map(lambda x: x.year)).mean().round()
recession['year'] = recession.index

# Create index for recession years and for growth years
rec = [i for i in range(len(recession.index)) if recession.iloc[i,0] == 1]
grow = [i for i in range(len(recession.index)) if recession.iloc[i,0] == 0]

# Compute cosine similarities between all documents, and subset to get those within recession years, 
# within growth years, and between recession and growth years
from sklearn.metrics.pairwise import cosine_similarity
cos = cosine_similarity(tf_idf)
cosine_rec = cos[rec,:][:,rec]
cosine_grow = cos[grow,:][:, grow]
cosine_cross = cos[rec,:][:, grow]

# Print result
print('cosine similarity within recession years', np.mean(cosine_rec))
print('cosine similarity within growth years', np.mean(cosine_grow))
print('cosine similarity between recession and growth years', np.mean(cosine_cross))

# Start again with tf_idf_hat
cos_hat = cosine_similarity(tf_idf_hat)
cosine_rec_hat = cos_hat[rec,:][:,rec]
cosine_grow_hat = cos[grow,:][:, grow]
cosine_cross_hat = cos[rec,:][:, grow]

# Print result
print('cosine similarity within recession years', np.mean(cosine_rec_hat))
print('cosine similarity within growth years', np.mean(cosine_grow_hat))
print('cosine similarity between recession and growth years', np.mean(cosine_cross_hat))

################## Program using EM Algorithm ####################






