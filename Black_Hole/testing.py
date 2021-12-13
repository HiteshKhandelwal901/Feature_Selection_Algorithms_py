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

"""
X, y = make_multilabel_classification(n_samples = 300,n_features = 4,sparse = True, n_labels = 3,
return_indicator = 'sparse', allow_unlabeled = False)
X = pd.DataFrame(X.toarray())
y = y.toarray()
print("X SHAPE  = ", X.shape,"type = ", type(X))
print("Y shape  = ", y.shape)
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
print("Y = \n\n", Y)

X = data_updated.drop(columns = Y)
print("X = \n\n", X)

print("INFO : \n\n")
print("X shape : ", X.shape)
print("X type = ", type(X))
print("Y shape = : ", Y.shape)
print("Y type: ", type(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)



#clf = BinaryRelevance(LogisticRegression())
#clf = BinaryRelevance(classifier = SVC())
clf = BinaryRelevance(classifier = RandomForestClassifier())
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test).toarray()
#print("y_pred = ", y_pred, "type = ", type(y_pred), "y pred shape = ", y_pred.shape)
#print("y_test = ", Y_test, "type = ", type(Y_test), "y test shape = ", Y_test.shape)
Y_test = Y_test.to_numpy()
print("total test labels = ", len(Y_test)*Y_test.shape[1])
print("total test samples = ", X_test.shape)
#print("y_test = ", Y_test, "type = ", type(Y_test), "y test shape = ", Y_test.shape)
score, correct, incorrect = hamming_get_accuracy(y_pred, Y_test)
print("train test split with binary relevance info :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
print("SCORE : ", score)
print("CORRECT : ", correct)
print("INCORRECT : ", incorrect)

#score,clf2, correct, incorrect = hamming_scoreCV(X_train,Y_train)


#y_pred = clf2.predict(X_test).toarray()

#score, correct, incorrect = hamming_get_accuracy(y_pred, Y_test)
#print("Hamming accuracy info :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
#print("SCORE : ", score)
#print("CORRECT : ", correct)
#print("INCORRECT : ", incorrect)






