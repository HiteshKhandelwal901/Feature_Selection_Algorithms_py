import pandas as pd
from scipy.stats import pearsonr
from utility import get_max_label_correlations_gen, get_max_corr_label
from sklearn.feature_selection import VarianceThreshold


def feature_correlation(X):
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
        #result  = sum(dataCorr[0])
        return dataCorr
        #print("sum = ", result)

    else:
        #print("result = ",0)
        return 0

    return dataCorr

if __name__ == "__main__":

    #step 1: load the data and preprocess it 
    data = pd.read_csv('Amino_MultiLabel_Dataset.csv') 
    print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
    print("data without header \n\n", data)
    column_names = []
    for i in range(data.shape[1]):
        column_names.append(str(i))
    #print("done assigning column names", column_names)

    data_updated = pd.read_csv('Amino_MultiLabel_Dataset.csv', names = column_names)

    Y = data_updated[['20','21','22','23']]
    print("Y = \n\n", Y)

    X = data_updated.drop(columns = Y)
    print("X = \n\n", X)

    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))


#step 2 : Caclulate the correlation matrix

corr_df = feature_correlation(X)
print("corr within attributes \n\n", corr_df)
#print("correlation matrix before= ", corr_df)
#corr_df[corr_df[0] <0.99]
#print("correlation matrix after= ", corr_df)

label_dict = get_max_label_correlations_gen(X,Y)
#print("each atribute max corr with label\n", label_dict)
result = get_max_corr_label(X, label_dict)
print("-----testing ----")
selector = VarianceThreshold(0.95)
vt = selector.fit(X)
#X[:, vt.variances_ > 0.95]
#print(X)
import numpy as np
#print("variances = ", vt.variances_)
idx = np.where(vt.variances_ > 50)[0]
#print(idx)
#print("X_removed shape = ", X_removed.shape)
#print("X_removed = ", X_removed)



#step 3 : if correlation > 0.99 add it to the list

corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

print("upper = ", upper)
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.095)]

print("to drop = ", to_drop)

    
# Drop features 
X = X.drop(X[to_drop], axis=1)

print("X shape = ", X.shape)

print(X)