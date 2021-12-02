from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)

#data = pd.read_csv('Iris.csv')
#Y = data['Species']
#X = data.drop(columns= ['Species', 'Id'])

for i in range(4,6):
    print("features number  = ", i)

    rfe = RFE(estimator=LinearSVC(), n_features_to_select=i)
    print("check 1")
    rfe.fit(X, y)
    print("check 2")
    x = rfe.transform(X)
    print("check 3")
    print("shape = ", x.shape)

    model = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



    