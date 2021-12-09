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


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


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