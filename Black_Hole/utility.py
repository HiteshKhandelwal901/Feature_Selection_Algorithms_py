from collections import defaultdict
import sklearn
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
from scipy.stats import pearsonr
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from skmultilearn.adapt import MLkNN
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score, coverage_error

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def ranking_loss(y_pred, y_true):
    return label_ranking_loss(y_true, y_pred)

def avg_precision_loss(y_pred, y_true):
    return label_ranking_average_precision_score(y_true, y_pred)

def coverage_error(y_pred, y_true):
    return coverage_error(y_true, y_pred)


def get_index_sum(X, cols):
    index_sum = []
    for col in cols:
        print("col = ", col)
        index_sum.append(X.columns.get_loc(col))
        print("index sum = ", index_sum)
    return sum(index_sum)

def get_max_label_correlations(X,Y):
    #print("INFO : X shape = ", X.shape)
    #print("cols  = ", X.columns)
    Y = pd.DataFrame(Y)
    """
    X -> dataframe
    Y -> array
    """
    max_corr = 0
    result = 0
    for cols in X.columns:
        max_corr = 0
        #print("cols = ", cols)
        x = X[cols]
        for label in Y.columns:
            #print("label = ", label)
            y = Y[label]
            corr, _ = pearsonr(x, y)
            corr = abs(corr)
            #print("corr = ", corr)
            if corr > max_corr:
                max_corr = corr
        result = result + max_corr
    #print("result = ", result)
    return result

def get_max_label_correlations_gen(X,Y):
    #print("INFO : X shape = ", X.shape)
    #print("cols  = ", X.columns)
    label_corr = defaultdict()
    Y = pd.DataFrame(Y)
    """
    X -> dataframe
    Y -> array
    """
    max_corr = 0
    result = 0
    for cols in X.columns:
        max_corr = 0
        #print("cols = ", cols)
        x = X[cols]
        for label in Y.columns:
            #print("label = ", label)
            y = Y[label]
            corr, _ = pearsonr(x, y)
            corr = abs(corr)
            #print("corr = ", corr)
            if corr > max_corr:
                max_corr = corr
        label_corr[cols] = max_corr
    return label_corr

def get_max_corr_label(X,label_corr_dict):
    result = 0
    for cols in X:
        result = result + label_corr_dict[cols]
    return result




def weighted_label_correlations(X,Y):
    label_corr = defaultdict()
    for cols in X.columns:
        #print("col = ", cols)
        result = 0
        x = X[cols]
        for label in Y.columns:
            #print("label = ", label)
            y = Y[label]
            corr, _ = pearsonr(x, y)
            corr = abs(corr)
            #print("corr = ", corr)
            result = result + 0.25*corr
        #print("result = ", result)
        label_corr[cols] = result
    return label_corr

def weighted_label_correlations_70_30(X,Y):
    label_corr = defaultdict()
    
    for cols in X.columns:
        corr_list = []
        #print("col = ", cols)
        result = 0
        x = X[cols]
        for label in Y.columns:
            #print("label = ", label)
            y = Y[label]
            corr, _ = pearsonr(x, y)
            corr = abs(corr)
            #print("corr = ", corr)
            #result = result + 0.25*corr
        #print("result = ", result)
            corr_list.append(corr)
        #print("corr list = ", corr_list)
        highest = max(corr_list)
        #print("highrst = ", highest)
        corr_list.sort()
        #print("corr sorted = ", corr_list)
        sec_highest = corr_list[-2]
        #print("sorted corr = ", sec_highest)
        
        result = 0.7*highest + 0.3*sec_highest
        label_corr[cols] = result
    return label_corr

def feature_correlation_sum(X):
    """
    X is dataframe
    """
    #print("INFO X :", X.shape, type(X))
    if X.shape[1]>1:
        #print("inside if")
        
        dataCorr = X.corr(method='pearson').abs()
        #print("dataCorr = ", dataCorr)
        #print("--------- correlations ------\n\n", dataCorr)
        dataCorr = dataCorr[abs(dataCorr) >= 0].stack().reset_index()
        dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]
        #print("dataCorr = ", dataCorr)
    
        # filtering out lower/upper triangular duplicates 
        #if names of the columns are string then uncomment this line and use this instead
        #dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([ x['level_0'],x['level_1']])),axis=1)
        
        dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([  str(x['level_0'] ) , str(x['level_1'])   ])),axis=1)
        #print("dataCorr after ordered cols = ", dataCorr)
        dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
        dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
        #print("final dataCorr = ", dataCorr)
        #dataCorr.sort_values(by=[0], ascending=False).head(10)
        #print("dataCorr = ", dataCorr)
        #print("columns = ", dataCorr.columns)
        result  = sum(dataCorr[0])
        #print("sum = ", result)

    else:
        #print("result = ",0)
        return 0

    return result


def hamming_get_accuracy(y_pred,y_test):
    #print("y_pred = ", y_pred)
    #print("y_test = ", y_test)
    #print("trying with key 0\n\n")
    #print("y_p[0] = ", y_pred[0])
    #print("y_t[0] = ", y_test[0])
    correct = 0
    incorrect = 0
    size = len(y_pred)*len(y_pred[0])
    #print("size = ", size)
    #print("y shape = ", y_pred.shape)
    #print("size = ", size)
    #prnt("Dsd")
    for i in range(len(y_pred)):
        #print("i = ", i)
        y_p = y_pred[i]
        #print("y_p = ", y_p)
        y_t = y_test[i]
        for j in range(len(y_p)):
            if y_p[j] == y_t[j]:
                correct = correct+1
    incorrect = size - correct

    #print("correct prediction = ", correct)

    return correct/size, correct, incorrect


def hamming_score(X,y, metric = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    #clf = BinaryRelevance(classifier = RandomForestClassifier(random_state= 25))
    clf = BinaryRelevance(classifier = MLkNN(k=10))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test).toarray()
    loss = hamming_loss(y_pred, y_test)
    score = 1-loss
    if metric == True:
        rl_loss = ranking_loss(y_test,y_pred)
        avg_precision = avg_precision_loss(y_test, y_pred)
        #covg_error = coverage_error(y_pred, y_test)
        return loss, rl_loss, avg_precision
    return score, loss




def hamming_scoreCV(X, y, n_splits = 5, model_name = "Random_forest"):
    #print("INSIDE HAMMING SCORE CV RECIVED X = ", X.shape)
    kf = KFold(n_splits)

    #X = X.toarray()
    #y = y.toarray()
    X_train = X
    Y_train = y

    #X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)
    #print("Splitting the recieved X in hamming_Score_CV before fold run | X_train and X_test= ", X_train.shape, X_test.shape)
    # print("X train inside hamming function ", X_train.shape)
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
        y_test = y_test.to_numpy()
        #print("y test shape = ", y_test.shape)
        #clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
        if model_name == "Random_forest":
            #print("inside random forest")
            clf = BinaryRelevance(classifier = RandomForestClassifier(random_state= 12))
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test).toarray()
            score,correct,incorrect = hamming_get_accuracy(y_pred, y_test)
            scores.append(score)
        

        if model_name == "SVC":
            clf = BinaryRelevance(classifier = SVC())
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test).toarray()
            score,correct, incorrect = hamming_get_accuracy(y_pred, y_test)
            scores.append(score)

    return np.mean(scores),clf, correct, incorrect