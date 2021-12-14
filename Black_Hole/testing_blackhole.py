import pandas as pd
from collections import defaultdict
import warnings,os
import pandas as pd
import random
import math
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_multilabel_classification
from utility import hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations, get_index_sum, get_max_label_correlations_gen, get_max_corr_label
import sklearn


print("-----BENCHING ON DIPEPTIDE DATASET-------")
data = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv')



#print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
#print("data without header \n\n", data)
column_names = []
for i in range(data.shape[1]):
    column_names.append(str(i))
#print("done assigning column names", column_names)
data_updated = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv', names = column_names)
col = [2, 4, 5, 7, 8, 9, 12, 15, 16, 19, 23, 25, 26, 27, 30, 31, 35, 36, 38]

col_list = []

for c in col:
    col_list.append(str(c))

data_updated = data_updated.drop(columns = col_list)
print("data shape = \n", data_updated.shape)

Y = data_updated[['400','401','402','403']]
print("Y = \n\n", Y)

X = data_updated.drop(columns = Y)
print("X = \n\n", X)

print("INFO : \n\n")
print("X shape : ", X.shape)
print("X type = ", type(X))
print("Y shape = : ", Y.shape)
print("Y type: ", type(Y))


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)
accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
y_pred = clf.predict(X_test).toarray()
y_test = Y_test.to_numpy()
score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
print("Hamming accuracy info Random Forest:\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
print("SCORE : ", score)
print("CORRECT : ", correct)
print("INCORRECT : ", incorrect)
print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
