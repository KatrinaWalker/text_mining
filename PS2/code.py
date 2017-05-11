#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:31:59 2017

@author: k2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import lda
from collections import Counter
import scipy.sparse as ssp
import topicmodels as tpm
from nltk import word_tokenize
from numpy.random import dirichlet
import time

# Question 1


### GIBBS SAMPLER

data = pd.read_table("~/Documents/text_mining_course/text_mining/PS1/speech_data_extend.txt", encoding="utf-8") 
data = data.loc[data['year']>=1945]
data = data.reset_index()


def data_preparation(data):
    prep_data = data.apply(lambda row: #tokenize
                            word_tokenize(row['speech'].lower()), axis=1)
    stop_w=set(stopwords.words('english')) #stopwords
    for i in range(len(prep_data)): #non-alphanumeric characters
        prep_data[i] = [w for w in prep_data[i] if w not in stop_w and w.isalpha()]
    stemmer = PorterStemmer() #Create a stemmer object
    for i in range(len(prep_data)): #Stem the data
        prep_data[i] = [stemmer.stem(elem) for elem in prep_data[i]]
    unique_words = np.unique([word for doc in prep_data for word in doc]).tolist()
    return prep_data, unique_words


prep_data, unique_words = data_preparation(data)

#theta = dirichlet([alpha]*K,D)
#beta = dirichlet([eta]*K,V)

# Problem: generate Z_dn, a list of dimension d of lists, each one of different length
# (number of terms of each document) with values in k. So: topic allocation of word
# n in document d. To assign a new topic to each entry, we need to match term v
# in Beta_v (which will be stored in "unique_words") with word n in Z_dn (which
# will be stored in "prep_data").


### Auxiliar functions Z sample
def simulate(K, row):
    samples = np.random.multinomial(1,[1/K]*K,len(prep_data[row])).tolist()
    samples_correct = []
    for s in samples:
        samples_correct.append(s.index(1))
    return samples_correct

def N_count(Z_d, K):
    N_count_vector = []
    for k in range(K):
        N_count_vector.append(Z_d.count(k))
    return N_count_vector

### Sample for topic allocation
def sample_topic(Z, theta, beta):
    D = len(Z)
    for d in range(D):
        n = len(Z[d])
        for i in range(n):
            beta_v = beta[unique_words.index(prep_data[d][i])]
            probs = (theta[d,:]*beta_v)/np.sum(theta[d,:]*beta_v)
            Z[d][i] = np.random.multinomial(1, probs).tolist().index(1)
    return Z
#sample_topic(Z, theta, beta)

### Sample for theta
def sample_theta(Z,alpha,theta):
    D,K = theta.shape
    N = np.zeros((D,K))
    for d in range(D):
        #N[d,:] = np.unique(Z[d], return_counts=True)[1]
        N[d,:] = N_count(Z[d], K)
        theta[d,:] = dirichlet(N[d,:] + alpha)
    return theta

### Sample for beta

def sample_beta(Z,prep_data,eta,beta):
    K = beta.shape[1]
    M = np.zeros((K,V))
    #Generate M
    s = [i for sublist in prep_data for i in sublist ]
    z_s = [z for sublist in Z for z in sublist]
    for k in range(K):
        words = [s[i] for i in range(len(s)) if z_s[i] == k]
        counts = Counter(words)
        for v in range(len(unique_words)):
            if unique_words[v] in counts: M[k,v] = counts[unique_words[v]]
    #Generate beta
    for k in range(K):
        beta[:,k] = dirichlet(M[k,:] + eta)
    return beta

#beta = sample_beta(Z,prep_data,eta,beta)

def gibbs_sampler(n_iter,prep_data,alpha,eta,K):
    ## Initialize objects
    D = len(prep_data)
    theta = dirichlet([alpha]*K,D)
    beta = dirichlet([eta]*K,V)
    Z = prep_data.apply(lambda row: simulate(K,row))
    Z_dist = []
    theta_dist = []
    beta_dist = []
    for i in range(n_iter):
        print('Iteration nº:'+ str(i))
        start = time.time()
        Z = sample_topic(Z,theta,beta)
        theta = sample_theta(Z,alpha,theta)
        beta = sample_beta(Z,prep_data,eta,beta)
        Z_dist.append(Z)
        theta_dist.append(theta)
        beta_dist.append(beta)
        print('Duration:'+ str(time.time()-start))
    return Z_dist, theta_dist, beta_dist

### Initial parameters
#D = len(prep_data) #Number of documents
#V = len(unique_words)#Number of unique terms
#Z = prep_data.apply(lambda row: simulate(K,row)) #Z_dn
#N = np.zeros((D,K))
#M = np.zeros((K,V))

#Initial values (reference original paper)
K = 10 #Number of topics
alpha = 50/K
eta = 200/V

Z_2, theta_2, beta_2 = gibbs_sampler(5000,prep_data, alpha, eta, K)




# Question 2



# Import the original data
data_origin = pd.read_table("~/Documents/text_mining_course/text_mining/PS1/speech_data_extend.txt", encoding="utf-8") 
dt_matrix = pd.DataFrame.from_csv("~/Documents/dt_matrix.csv", sep = ";", index_col = 0) 

X = dt_matrix.iloc[:][data_origin.year >= 1945]
X = ssp.csr_matrix(X.astype(int))

K = 2

Col_Gibbs = tpm.LDA.LDAGibbs(prep_data,K)

Col_Gibbs.alpha
Col_Gibbs.beta

burn_samples = 1000
jumps = 50
used_samples = 10

Col_Gibbs.sample(burn_samples,jumps,used_samples)

Col_Gibbs.perplexity() 

word_topics = Col_Gibbs.tt 
doc_topics = Col_Gibbs.dt 



# Question 3

# Select only documents which appear after 1945
X = dt_matrix.iloc[:][data_origin.year >= 1945]
X = X.reset_index()
X = X.drop('level_0', axis=1)

# Create a new variable with only presidents and years
parties = data_origin.loc[:,['president', 'year']]
parties = parties.reset_index()

# Create a new variable with 1 when presidents are Democrats, 0 when they are Republicans
zero_len = pd.Series(np.zeros(len(parties.index)))
parties['parties'] = zero_len
parties['parties'] = (parties.president == "Truman") | (parties.president == "Kennedy") | (parties.president == "Johnson") | (parties.president == "Carter") | (parties.president == "Clinton") | (parties.president == "Obama")
parties.parties = list(map(lambda x: 1 if x else 0, parties.parties))
parties = parties.iloc[:][parties.year >= 1945]

# Set the parties variable as y
y = parties.parties

# Split the X and y variables into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y, test_size = 0.3, random_state = 42)

# Create the logistic regression estimator with an l1 loss parameter
log_reg = LogisticRegression(penalty = "l1")

# Set some parameters to tune over
c_space = [1.3, 1.5, 1.7]
parameters = {'C': c_space}

# Create a cross-validation estimator
cv = GridSearchCV(log_reg, parameters)

# Fit the model, predict and report the accuracy and the best parameter
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))




## Run a logistic regression using the topics instead of the document-term matrix
X = dt_matrix.iloc[:][data_origin.year >= 1945]
X = ssp.csr_matrix(X.astype(int))

K, S, alpha, eta = 20, 1000, 0.1, 0.01

Col_Gibbs = lda.LDA(n_topics=K, n_iter=S, alpha=alpha, eta=eta)

X = Col_Gibbs.fit_transform(X)

# Split the X and y variables into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 59)

# Create the logistic regression estimator
log_reg = LogisticRegression("l2")

# Set some parameters to tune over
c_space = [0.5, 2, 5]
parameters = {'C': c_space}

# Create a cross-validation estimator
cv = GridSearchCV(log_reg, parameters)

# Fit the model, predict and report the accuracy and the best parameter
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))








