import pandas as pd
from scipy.stats import pearsonr


def feature_correlation(X):
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
print("correlation matrix = ", corr_df)


#step 3 : if correlation > 0.99 add it to the list
