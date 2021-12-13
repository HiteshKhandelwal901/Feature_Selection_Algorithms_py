from collections import defaultdict
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from utility import hamming_scoreCV, hamming_get_accuracy,feature_correlation_sum, get_max_label_correlations
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import math

"""
X, y = make_multilabel_classification(n_samples = 300,n_features = 4,sparse = True, n_labels = 3,
return_indicator = 'sparse', allow_unlabeled = False)
X = pd.DataFrame(X.toarray())
y = y.toarray()
print("X SHAPE  = ", X.shape,"type = ", type(X))
print("Y shape  = ", y.shape)
"""

def hamming_scoreCV_test(X,y,n_splits = 5, model_name = "Random_forest"):
    kf = KFold(n_splits)

    #X = X.toarray()
    #y = y.toarray()
    #print("inside ham test")

    if model_name == "SVC":
        parameter_dict = defaultdict()
        parameter_dict['C'] = [1, 10, 100, 1000, 10000]
        parameter_dict['sigma'] = [0.01, 0.1, 10,100, 10000]

    if model_name == "Random_forest":
        global_max_score = 0
        selection_list = []
        percent_list = [0.2,0.4,0.6,0.8, 1]
        for i in percent_list:
            #print("curr i = ", i)
            #print("per  = ", i*X.shape[1])
            selection_list.append(math.ceil(i*X.shape[1]))
        print("selection list = ", selection_list)
        parameter_dict = defaultdict()
        parameter_dict['depth'] = [50,100,200,500]
        parameter_dict['Mtry'] = selection_list

        for depth in parameter_dict['depth']:
            #print("depth = ", depth)
            for mtry in parameter_dict['Mtry']:
                #print("mtry  = ", mtry)

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
                    y_test = Y_test.to_numpy()
                    #print("y test shape = ", y_test.shape)
                    #clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
                  
                    clf = BinaryRelevance(classifier = RandomForestClassifier(n_estimators = depth, max_features = mtry))
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test).toarray()
                    #print("type of y_pred = ", type(y_pred))
                    #print("y_pred = ", y_pred)
                    #print("type after conversion = ", y_pred.toarray())
                    score,correct, incorrect = hamming_get_accuracy(y_pred, y_test)
                    scores.append(score)
                print("-----INFORMATION ---------\n\n")
                print("max_trees = ", depth)
                print("max features = ", mtry)
                print("random forest score = ", np.mean(scores))
                if np.mean(scores) > global_max_score:
                    global_max_score = np.mean(scores)
                    opt_max_trees = depth
                    opt_max_features = mtry
                    print("global max = ", global_max_score)
                    print("\n\n")
        print("global max score = ", global_max_score)
        print("optimal max tree = ", opt_max_trees)
        print("optimal max features = ", opt_max_features)
    return np.mean(scores),clf, correct, incorrect
        

    



            

    
    """"
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
        y_test = Y_test.to_numpy()
        #print("y test shape = ", y_test.shape)
        #clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
        if model_name == "Random_forest":
            clf = BinaryRelevance(classifier = RandomForestClassifier())
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test).toarray()
            #print("type of y_pred = ", type(y_pred))
            #print("y_pred = ", y_pred)
            #print("type after conversion = ", y_pred.toarray())
            score,correct, incorrect = hamming_get_accuracy(y_pred, y_test)
            scores.append(score)
    return np.mean(scores),clf, correct, incorrect
    """
    







data = pd.read_csv('Amino_MultiLabel_Dataset.csv') 
column_names = []
for i in range(data.shape[1]):
    column_names.append(str(i))
#print("done assigning column names", column_names)

data_updated = pd.read_csv('Amino_MultiLabel_Dataset.csv', names = column_names)
#print("done adding headers \n")
#print("data updated = \n\n", data_updated)
#print("data columns = ", data_updated.columns)
#print("testing columns")

#print("data[1] = \n\n", data_updated['1'])

Y = data_updated[['20','21','22','23']]
#print("Y = \n\n", Y)

X = data_updated.drop(columns = Y)
#print("X = \n\n", X)

#X = X[['0', '1', '2','3']]



print("INFO : \n\n")
print("X shape : ", X.shape)
print("X type = ", type(X))
print("Y shape = : ", Y.shape)
print("Y type: ", type(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)


hamming_scoreCV_test(X_train, Y_train)

#"""
#clf = BinaryRelevance(LogisticRegression())
#clf = BinaryRelevance(classifier = SVC())
#clf = BinaryRelevance(classifier = RandomForestClassifier())
#clf.fit(X_train, Y_train)
#y_pred = clf.predict(X_test).toarray()
#print("y_pred = ", y_pred, "type = ", type(y_pred), "y pred shape = ", y_pred.shape)
#print("y_test = ", Y_test, "type = ", type(Y_test), "y test shape = ", Y_test.shape)
#Y_test = Y_test.to_numpy()
#print("total test labels = ", len(Y_test)*Y_test.shape[1])
#print("total test samples = ", X_test.shape)
#print("y_test = ", Y_test, "type = ", type(Y_test), "y test shape = ", Y_test.shape)
#score, correct, incorrect = hamming_get_accuracy(y_pred, Y_test)
#print("train test split with binary relevance info :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
#print("SCORE : ", score)
#print("CORRECT : ", correct)
#print("INCORRECT : ", incorrect)

#score,clf2, correct, incorrect = hamming_scoreCV(X_train,Y_train)


#y_pred = clf2.predict(X_test).toarray()

#score, correct, incorrect = hamming_get_accuracy(y_pred, Y_test)
#print("Hamming accuracy info :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
#print("SCORE : ", score)
#print("CORRECT : ", correct)
#print("INCORRECT : ", incorrect)
#"""




