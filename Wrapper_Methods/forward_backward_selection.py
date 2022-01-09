from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import pandas as pd

def preprocess_data(data):
    #drop label and convert label from non numerical to numerical
    Y = data['Species']
    X = data.drop(columns= ['Species', 'Id'])
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y)
   
    #Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)
    return X_train, X_test, Y_train, Y_test

def read_data(name):
    df = pd.read_csv(name)
    return df

def forward_selection(data, label, k_features=1):
    lreg = LinearRegression()
    sfs1 = sfs(lreg,k_features, forward=True, verbose=0, scoring='neg_mean_squared_error')
    sfs1 = sfs1.fit(data, label)
    return sfs1

def backward_selection(data, label,k_features=1):
    lreg = LinearRegression()
    sfs2 = sfs(lreg, k_features, forward=False, verbose=2, scoring='neg_mean_squared_error')
    sfs2 = sfs2.fit(data, label)
    return sfs2

    




if __name__ == "__main__":
    #read the input data
    data = read_data('Iris.csv')

    #preprocess and split the data
    X_train, X_test, Y_train, Y_test = preprocess_data(data)
    
    n_features = X_train.shape[1]
    feature_dict = defaultdict()
    backward_dict = defaultdict()


    for n in range(1, n_features+1):
        f =  forward_selection(X_train, Y_train, n)
        feature_names = list(f.k_feature_names_)
        score = f.k_score_
        feature_dict[-1*score] = feature_names
    
    
    final_score = max(feature_dict)
    final_feat = feature_dict[final_score]
    print("feature dict = ", feature_dict)
    print("best subset = ", final_feat, "with score = ", final_score )
    



