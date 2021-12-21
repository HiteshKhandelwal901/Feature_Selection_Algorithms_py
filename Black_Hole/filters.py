from collections import defaultdict
from numpy.core.fromnumeric import var
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from operator import itemgetter


def corr_features(X):
    dataCorr = X.corr(method='pearson').abs()
    dataCorr = dataCorr[abs(dataCorr) >= 0].stack().reset_index()
    dataCorr = dataCorr[dataCorr['level_0'].astype(str)!=dataCorr['level_1'].astype(str)]
    dataCorr['ordered-cols'] = dataCorr.apply(lambda x: '-'.join(sorted([  str(x['level_0'] ) , str(x['level_1'])   ])),axis=1)
    dataCorr = dataCorr.drop_duplicates(['ordered-cols'])
    dataCorr.drop(['ordered-cols'], axis=1, inplace=True)
    
    return dataCorr

def Var_features(X):
    names = defaultdict()
    var_series = X.var()
    for index,cols in enumerate(X.columns):
        names[cols] = var_series[index]
    return names

def get_least_var(columns, var_dict):
    if var_dict[columns[0]] <= var_dict[columns[1]]:
        return columns[0]
    else:
        return columns[1]


def remove_features(X):
    names = Var_features(X)
    columns_drop = get_col_to_drop(X,names)
    print("columns to drop = ", columns_drop)
    X = X.drop(columns = columns_drop)
    return X

def get_col_to_drop(X, names):
    corr_df = corr_features(X)
    column_drop = []
    for i in range(corr_df.shape[0]):
        columns = []
        for j in range(corr_df.shape[1]):

            if j<2:

                columns.append(corr_df.iloc[i,j])
            else:

                
                if corr_df.iloc[i,j]>0.95:
                    
                    cols = get_least_var(columns, names)
                    
                    column_drop.append(cols)
        
    
        

    return set(column_drop)

def remove_features_with_low_variance(X):
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(X)
    return X

def univariate_feature_elimination(X,y, k):
    sk = SelectKBest(score_func=chi2, k=4)
    X_new = sk.fit_transform(X,y)
    #print("X = ",X)
    scores = sk.scores_
    #print("score = \n", scores, "type = ", type(scores), "shape = ", scores.shape)
    columns_to_drop = get_least_columns(X, scores,k)
    X = X.drop(columns = columns_to_drop)
    return X

def get_least_columns(X, scores,k):
    columns_to_remove = []
    names = defaultdict()
    for index,col in enumerate(X.columns):
        names[col] = scores[index]
    remove_dic = dict(sorted(names.items(), key = itemgetter(1))[:k])
    #print("remove dic = \n", remove_dic)

    for keys in remove_dic:
        columns_to_remove.append(keys)
    return columns_to_remove
        




    





if __name__ == "__main__":
    data = pd.read_csv('scene.csv')
    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    X = remove_features(X)
    print("after removing we have X = ", X)



        
    


