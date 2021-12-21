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
from utility import weighted_label_correlations,hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations, get_index_sum, get_max_label_correlations_gen, get_max_corr_label
import sklearn
from filters import remove_features,univariate_feature_elimination



if __name__ == "__main__":

    
    data = pd.read_csv("scene.csv")
    print("data = \n", data)
    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    print("X = \n\n", X)

    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))
    k_list = [5,10,15,20,25,30,35,40,45,50]
    k_list2 = [15]
    for k in k_list2:

        X = univariate_feature_elimination(X,Y,5)
        #X = remove_features(X)



        print("removed least variance of highly correlared feature\n\n")
        print("\n\n-----without feature selection ----- \n\n")
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=42)
        accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
        y_pred = clf.predict(X_test).toarray()
        y_test = Y_test.to_numpy()
        score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
        print("K = || ", k)
        print("Hamming score info for without feature selection :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
        print("SCORE : ", score)
        print("CORRECT : ", correct)
        print("INCORRECT : ", incorrect)
        print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )