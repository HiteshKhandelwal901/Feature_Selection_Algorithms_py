import pandas as pd
from collections import defaultdict
import warnings
import os
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
from filters import remove_features, univariate_feature_elimination


print("-----BENCHING ON DIPEPTIDE DATASET-------")
data = pd.read_csv('yeast_clean.csv')


#print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
#print("data without header \n\n", data)
#column_names = []

# for i in range(data.shape[1]):
#    column_names.append(str(i))
#print("done assigning column names", column_names)
#data_updated = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv', names = column_names)


index_to_names = defaultdict()
#feature_index = []


#data_updated = data_updated.drop(columns = col_list)
#print("data shape = \n", data_updated.shape)

Y = data[['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10','Class11','Class12','Class13','Class14']]
X = data.drop(columns= Y)
scaled_features = sklearn.preprocessing.MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(scaled_features, index= X.index, columns= X.columns)
X = univariate_feature_elimination(X, Y, 15)


print("INFO : \n\n")
print("X shape : ", X.shape)
print("X type = ", type(X))
print("Y shape = : ", Y.shape)
print("Y type: ", type(Y))


print("----- with feature subset ------")

for index, col in enumerate(X.columns):
    index_to_names[index] = col

#print("index deict", index_to_names)

column_names = []
col_list =   [2, 13, 18, 21, 28, 29, 31, 34, 42, 43, 46, 49, 50, 57, 61, 64, 65, 69, 78, 79, 81]


for each in col_list:
    #print("each = ", each)
    column_names.append(index_to_names[each])

#print("column_names = ", column_names)

X = X.drop(columns=column_names)

print("subset shape = ", X.shape)

df = pd.concat((X, Y), axis=1)
df.to_csv('subset_data/standalone_yeast_68.csv')


score, clf, correct, incorrect = hamming_scoreCV(X, Y)
print("ham loss = ", 1-score)
