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
col = [0, 1, 4, 6, 7, 8, 10, 13, 14, 15, 17, 20, 21, 22, 26, 27, 31, 32, 37, 42, 43, 47, 51, 59, 60, 61, 62, 63, 65, 73, 76, 78, 88, 90, 92, 94, 95, 96, 99, 102, 106, 109, 118, 123, 124, 125, 126, 127, 130, 131, 132, 133, 135, 139, 143, 146, 147, 150, 152, 153, 158, 164, 166, 167, 168, 174, 176, 179, 180, 182, 183, 184, 188, 190, 191, 193, 194, 195, 198, 209, 212, 214, 216, 223, 227, 228, 229, 237, 238, 240, 246, 249, 252, 253, 254, 255, 257, 258, 261, 264, 271, 277, 281, 284, 288, 290, 293]

col_list = []

for c in col:
    col_list.append(str(c))

data_updated = data_updated.drop(columns = col_list)
print("data shape = \n", data_updated.shape)

Y = data_updated[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
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
