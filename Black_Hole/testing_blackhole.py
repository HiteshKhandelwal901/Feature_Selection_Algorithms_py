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
data = pd.read_csv('scene.csv')



#print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
#print("data without header \n\n", data)
#column_names = []

#for i in range(data.shape[1]):
#    column_names.append(str(i))
#print("done assigning column names", column_names)
#data_updated = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv', names = column_names)



index_to_names = defaultdict()
#feature_index = []





#data_updated = data_updated.drop(columns = col_list)
#print("data shape = \n", data_updated.shape)

Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
print("Y = \n\n", Y)

X = data.drop(columns = Y)
print("X = \n\n", X)

print("INFO : \n\n")
print("X shape : ", X.shape)
print("X type = ", type(X))
print("Y shape = : ", Y.shape)
print("Y type: ", type(Y))


print("---- with complete dataset -------")

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



print("----- with feature subset ------")

for index,col in enumerate(X.columns):
    index_to_names[index] = col

#print("index deict", index_to_names)

column_names = []

col_list = [0, 2, 4, 8, 9, 12, 14, 15, 16, 17, 18, 21, 26, 29, 30, 31, 32, 35, 37, 40, 41, 43, 45, 47, 49, 52, 59, 64, 67, 72, 73, 74, 75, 76, 79, 82, 83, 85, 86, 90, 95, 97, 98, 100, 104, 114, 116, 117, 119, 120, 122, 123, 126, 130, 132, 140, 141, 144, 145, 148, 151, 156, 157, 159, 160, 161, 163, 165, 166, 167, 168, 170, 171, 172, 178, 180, 182, 183, 184, 185, 187, 188, 189, 190, 192, 193, 195, 196, 203, 204, 208, 209, 210, 211, 213, 218, 219, 225, 226, 227, 230, 231, 233, 234, 237, 240, 245, 249, 251, 255, 256, 257, 258, 259, 261, 264, 265, 267, 271, 275, 277, 278, 282, 284, 285, 291, 293]

for each in col_list:
    #print("each = ", each)
    column_names.append(index_to_names[each])

print("column_names = ", column_names)

X = X.drop(columns= column_names)

print("subset shape = ", X.shape)


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
