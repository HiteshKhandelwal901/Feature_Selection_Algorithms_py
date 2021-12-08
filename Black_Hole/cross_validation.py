from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import sys
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier




if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def hamming_get_accuracy(y_pred,y_test):
    correct = 0
    size = len(y_pred)*len(y_pred[0])
    for i in range(len(y_pred)):
        #print("i = ", i)
        y_p = y_pred[i]
        #print("y_p = ", y_p)
        y_t = y_test[i]
        #print("y_t = ", y_t)
        #print("length of y_p = ", len(y_p))
        for j in range(len(y_p)):
            #print("j = ", j)
            #print("y_p[j] = ", y_p[j])
            if y_p[j] == y_t[j]:
                correct = correct+1
        #print("correct labels = ", correct)
        #print("size = ", size)
        #print("corect/size = ", correct/size)
    return correct/size

#y = [[0,0,1,1], [1,0,1,0], [0,0,1,1]]
#y_true = [[0,0,1,1], [0,0,1,0], [0,1,0,1]]

#print(hamming_get_accuracy(y, y_true))








"""
data = pd.read_csv('Iris.csv')
Y = data['Species']
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)
X = data.drop(columns= ['Species', 'Id'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)
print("X TRAIN SHAPE = ", X_train.shape)
print("X test shape = ", X_test.shape)
"""



#strtfdKFold = StratifiedKFold(n_splits=5)
kf = KFold(n_splits=3)
X, y = make_multilabel_classification(n_features = 3,sparse = True, n_labels = 3,
    return_indicator = 'sparse', allow_unlabeled = False)
print("X SHAPE  = ", X.shape)
print("Y shape  = ", y.shape)

X = X.toarray()
y = y.toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)



kfold = kf.split(X_train, Y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    print("K = ", k)
    #x_train = X_train.iloc[train, :]
    x_train = np.take(X_train, train , axis = 0)
    #print("x_train shape = ", x_train.shape)
    #y_train = y_train.iloc[train]
    y_train = np.take(Y_train, train , axis = 0)
    #print("y_train shape = ", y_train.shape)
    #x_test = X_train.iloc[test, :]
    x_test = np.take(X_train, test , axis = 0)
    #print("x_test shape = ", x_test.shape)
    #y_test = y_train.iloc[test]
    y_test = np.take(Y_train, test , axis = 0)
    #print("y test shape = ", y_test.shape)
    #LR = LogisticRegression(max_iter = 1000, verbose = 0).fit(x_train, y_train)
    clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = hamming_get_accuracy(y_pred, y_test)
    #score = clf.score(x_test,y_test)
    print("in folf {} score  = {}".format(k, score))
    scores.append(score)
    
    



print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
