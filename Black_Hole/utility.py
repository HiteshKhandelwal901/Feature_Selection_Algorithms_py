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
        y_p = y_pred[i]
        y_t = y_test[i]
        for j in range(len(y_p)):
            if y_p[j] == y_t[j]:
                correct = correct+1
    return correct/size

def hamming_scoreCV(X, y, n_splits = 5):
    kf = KFold(n_splits)

    #X = X.toarray()
    #y = y.toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)
    
    kfold = kf.split(X_train, Y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        #print("K  = ", k)
        x_train = np.take(X_train, train , axis = 0)
        #print("x_train shape = ", x_train.shape)
        y_train = np.take(Y_train, train , axis = 0)
        #print("y_train shape = ", y_train.shape)
        x_test = np.take(X_train, test , axis = 0)
        #print("x_test shape = ", x_test.shape)
        y_test = np.take(Y_train, test , axis = 0)
        #print("y test shape = ", y_test.shape)
        clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = hamming_get_accuracy(y_pred, y_test)
        scores.append(score)
    return np.mean(scores)