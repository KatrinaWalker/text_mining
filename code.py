#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:00:14 2017

@author: davidrosenfeld
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]

sample = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/sample.dat")

print(sample)

# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

# Create a neighbors variable from 1 to 10 and some empty
# vectors for the train and test accuracy, for each different number of k-neighbour rule.
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Run our knn classifier over each value of k and obtain a training and test score.
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(X_train, y_train)
    
    train_accuracy[k-1] = knn.score(X_train, y_train)
    test_accuracy[k-1] = knn.score(X_test, y_test)
    
# Plot the training and test accuracy for each value of k
plt.plot(neighbors, train_accuracy, label = "train accuracy")
plt.plot(neighbors, test_accuracy, label = "test accuracy")
plt.title("Train and test accuracy")
plt.legend()
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.show()


# vectors for the train and test accuracy, for each different number of k-neighbour rule.
neighbors = np.arange(1, 20)
test_accuracy = np.empty(len(neighbors))

for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors = k)
    
    test_accuracy[k-1] = np.mean(cross_val_score(knn, X, y, cv = 10))

plt.plot(neighbors, test_accuracy, label = "test accuracy")
plt.title("Test accuracy")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.show()


# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

# Create a 9-NN classifier. Fit it on the data, and use it to make predictions
knn = KNeighborsClassifier(n_neighbors = 9)

knn.fit(X, y)
prediction = knn.predict(X_forecast_nolabel)
prediction = pd.DataFrame(prediction)

# Merge the ID and prediction dataframes. Rename them to fit the requested format.
forecast = pd.concat([forecast_id, prediction], axis = 1)
forecast.columns = ['Id', 'Prediction']
forecast.info()

# Save the prediction
forecast.to_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/forecast.csv", sep = ",", index=False)


# Create a 8-NN classifier. Fit it on the data, and use it to make predictions
knn = KNeighborsClassifier(n_neighbors = 8)

knn.fit(X, y)
prediction = knn.predict(X_forecast_nolabel)
prediction = pd.DataFrame(prediction)

# Merge the ID and prediction dataframes. Rename them to fit the requested format.
forecast = pd.concat([forecast_id, prediction], axis = 1)
forecast.columns = ['Id', 'Prediction']
forecast.info()

# Save the prediction
forecast.to_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/forecast2.csv", sep = ",", index=False)

forecast.info()
forecast.head(30)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]


steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors': [8, 9, 10]}

# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)              

cv = GridSearchCV(pipeline, parameters)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors': [8, 9]}

cv = GridSearchCV(pipeline, parameters, cv = 4)

cv.fit(X, y)

# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

prediction = cv.predict(X_forecast_nolabel)



# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

prediction = cv.predict(X_forecast_nolabel)
prediction = pd.DataFrame(prediction)

# Merge the ID and prediction dataframes. Rename them to fit the requested format.
forecast = pd.concat([forecast_id, prediction], axis = 1)
forecast.columns = ['Id', 'Prediction']
forecast.info()

# Save the prediction
forecast.to_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/forecast3.csv", sep = ",", index=False)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
X.info()
X.describe()
sum(X["47"] == 999)

X.iloc[:,58:78].plot(kind = 'box')
plt.show()


X["24"].plot(kind = 'hist', bins = 50)
plt.show()

30, 56
21, 22, 23, 45, 46, 47

X.iloc[:,19].plot(kind = 'hist', bins = 50)
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer

#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]

# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

# Replace values 9999 and 999 by 'NaN' in both X and X_forecast_nolabel
X[X == 9999] = np.nan
X[X == 999] = np.nan

print(X.isnull().sum())

X_forecast_nolabel[X_forecast_nolabel == 9999] = np.nan
X_forecast_nolabel[X_forecast_nolabel == 999] = np.nan

print(X_forecast_nolabel.isnull().sum())

# Set up an imputation method which we will add to the pipeline
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)

steps = [("imputation", imp), ("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors': [8, 9]}


# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

cv = GridSearchCV(pipeline, parameters, cv = 4)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##### Redo imputation using most frequent value instead of mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer

#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]

# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

# Replace values 9999 and 999 by 'NaN' in both X and X_forecast_nolabel
X[X == 9999] = np.nan
X[X == 999] = np.nan

print(X.isnull().sum())

X_forecast_nolabel[X_forecast_nolabel == 9999] = np.nan
X_forecast_nolabel[X_forecast_nolabel == 999] = np.nan

print(X_forecast_nolabel.isnull().sum())

# Set up an imputation method which we will add to the pipeline
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)

steps = [("imputation", imp), ("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors': [8, 9]}


# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

cv = GridSearchCV(pipeline, parameters, cv = 4)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Best score is {}".format(cv.best_score_))




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##### Redo imputation using most frequent value instead of mean


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression


#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]

# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

# Replace values 9999 and 999 by 'NaN' in both X and X_forecast_nolabel
X[X == 9999] = np.nan
X[X == 999] = np.nan

X_forecast_nolabel[X_forecast_nolabel == 9999] = np.nan
X_forecast_nolabel[X_forecast_nolabel == 999] = np.nan

# Set up an imputation method which we will add to the pipeline
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)

# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

steps = [("imputation", imp), ("scaler", StandardScaler()), ("logreg", LogisticRegression(penalty = 'l1'))]

pipeline = Pipeline(steps)

c_space = [0.02, 0.03, 0.05]
parameters = {'logreg__C': c_space}

cv = GridSearchCV(pipeline, parameters)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Best score is {}".format(cv.best_score_))

print(c_space)

# Run it on the whole data
cv.fit(X, y)

prediction = cv.predict(X_forecast_nolabel)
prediction = pd.DataFrame(prediction)

# Merge the ID and prediction dataframes. Rename them to fit the requested format.
forecast = pd.concat([forecast_id, prediction], axis = 1)
forecast.columns = ['Id', 'Prediction']
forecast.info()

# Save the prediction
forecast.to_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/forecast4.csv", sep = ",", index=False)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##### Use XGBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
import xgboost

#Import data. Remove the columns with the IDs
X_with_label = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_train.dat")
X = X_with_label.drop(X_with_label.columns[[0]], axis = 1)
X_forecast = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/X_test.dat")
y_with_id = pd.read_csv("/Users/davidrosenfeld/Documents/Kaggle_comp/y_train.dat", header = None, names = ["id", "label"])
y = y_with_id["label"]

# Extract only the variables for the X. Also create separate dataframe with only IDs
X_forecast_nolabel = X_forecast.drop(X_forecast.columns[[0]], axis = 1)
forecast_id = X_forecast[[0]]

# Replace values 9999 and 999 by 'NaN' in both X and X_forecast_nolabel
X[X == 9999] = np.nan
X[X == 999] = np.nan

print(X.isnull().sum())

X_forecast_nolabel[X_forecast_nolabel == 9999] = np.nan
X_forecast_nolabel[X_forecast_nolabel == 999] = np.nan

print(X_forecast_nolabel.isnull().sum())

# Set up an imputation method which we will add to the pipeline
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)

# Split the X and ys between training and test sets using sklearn.model_selection's train_test_split function.
# We define a size for the test set of 30% of the data. The set the seed of the randomiser at 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

steps = [("imputation", imp), ("scaler", StandardScaler()), ("logreg", LogisticRegression(penalty = 'l1'))]

pipeline = Pipeline(steps)































