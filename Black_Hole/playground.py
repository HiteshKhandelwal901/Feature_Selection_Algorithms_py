import pandas as pd
from scipy.stats import pearsonr
from sklearn import preprocessing
import numpy as np






def get_all_correlations(X):
    """
    X is dataframe
    """
    dataCorr = X.corr(method='pearson').abs()
    print("--------- correlations ------\n\n", dataCorr)
    dataCorr = dataCorr[abs(dataCorr) >= 0].stack().reset_index()
    dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]
    print("dataCorr = ", dataCorr)
    # filtering out lower/upper triangular duplicates 
    dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)
    print("dataCorr after ordered columns = ",dataCorr)
    dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
    dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
    
    #dataCorr.sort_values(by=[0], ascending=False).head(10)
    print("dataCorr = ", dataCorr)
    print("columns = ", dataCorr.columns)
    result  = sum(dataCorr[0])
    print("sum = ", result)
    return 

def get_max_label_correlations(X,Y):
    """
    X -> dataframe
    Y -> dataframe with atleast 2 labels
    """
    max_corr = 0
    result = 0
    for cols in X.columns:
        x = X[cols]
        for label in Y.columns:
            y = Y[label]
            corr, _ = pearsonr(x, y).abs()
            print("corr = ", corr)
            if corr > max_corr:
                max_corr = corr
        result = result + max_corr
        print("result = ", result)
        return result


data = pd.read_csv('Iris.csv')
Y = data['Species']
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)
#Y = Y.reshape(len(Y), 1)
#print("type(y) = ", type(Y))
X = data.drop(columns= ['Species', 'Id'])

result = get_all_correlations(X)
#result2 = get_max_label_correlations(X,Y)