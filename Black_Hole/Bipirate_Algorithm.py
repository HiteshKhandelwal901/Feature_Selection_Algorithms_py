from collections import defaultdict
import pandas as pd
import numpy as np

def get_data(filename, sample = True):
    if sample == True:
        feat_list = [[0.0347, 0.0897, 0.0912, 73.3024,6.2152],
                     [0.0814, 0.2727, 0.0857, 62.5844, 3.1832],
                     [0.1105, 0.2736, 0.0844, 65.2353, 2.7956]]
        label_list = [[0,1,1], [1,0,0], [0,1,0]]
        X = pd.DataFrame(feat_list, columns = ['0', '1', '2', '3', '4'])
        Y = pd.DataFrame(label_list, columns = ['0', '1', '2'])
        return X,Y

    data = pd.read_csv(filename) 
    column_names = []
    for i in range(data.shape[1]):
        column_names.append(str(i))
    
    data_updated = pd.read_csv(filename, names = column_names)

    Y = data_updated[['20','21','22','23']]

    X = data_updated.drop(columns = Y)
    return X,Y

def get_distance_corrlation_matrix(X,Y):
    #intialze the cordix matrix with rows = features and cols = labels
    cordix_matrix = np.zeros((X.shape[1], Y.shape[1]))
    for i,col in enumerate(X.columns):
        #for each feature
        x_i = X[col]
        for j,label in enumerate(Y.columns):
            #calculate cov matrix [4x4] for given feature with each label : (X1, Y1), (X1,Y2)..
            y_j = Y[label]
            #cov matrix
            cov_matr = np.cov(x_i,y_j)
            #cov value from the matrix
            cov_val = cov_matr[0][1]
            #get the var x from cov matrix
            var_X = cov_matr[0][0]
            #get the var y from the cov matrix
            var_Y = cov_matr[1][1]
            #denominator of the cordix eq
            deno = np.sqrt((var_X*var_Y))
            #get the final value
            value = cov_val / deno
            #subtract wiht 1
            CD_ij = 1- value
            #place the value in cordix matix's appropriate location
            cordix_matrix[i][j] = CD_ij
    return cordix_matrix

def get_label_counts(label):
    return list(label).count(1)

def get_label_weights(Y):
    size = Y.shape[0]
    weight = []
    for index,col in enumerate(Y):
        #for each column in Y, get the counts of 1 and divide by toal
        label = Y[col]
        counts = get_label_counts(label)
        weight.append(counts/size)
    return weight

def get_weighted_cordix(matr, weights):
    #multiply the matrix with the label weights
    return (matr * weights)


def euclid_dist(x,y):
    return np.linalg.norm(x-y)

def EDM(matr):
    ECD_matr = np.zeros((matr.shape[0], matr.shape[0]))
    for i in range(matr.shape[0]):
        #print(matr[i])
        #for each row
        for j in range(matr.shape[0]):
            #print(matr[j])
            #loop over all ther rows
            dist = euclid_dist(matr[i], matr[j])
            #apend the distcne to the ecd matrix
            ECD_matr[i][j] = dist
    return ECD_matr

def CDV(matr):
    cdv = []
    for row in range(matr.shape[0]):
        cdv.append(sum(matr[row]))
    return cdv

def get_row_sum(matr):
    return np.sum(matr)




def distance_correlation_dict_gen(X, Y):
    """
    Function to generate dicitonary mapping X's attributes to the sum of its weighted correlation distance with labels
    Ex :  X1 -> sum( (x1, y1), (x1, y2).....(x1, yn))

    Args:
    X - > Attr DF with headres 
    Y - > Label DF with headers

    Returns :
    Distance correlation dictionary
    
    """
    dist_corr = defaultdict()
    cordix_matr = get_distance_corrlation_matrix(X,Y)
    label_weights = get_label_weights(Y)
    Weighted_cordix = get_weighted_cordix(cordix_matr, label_weights)
    #print("weigthes_cordix = \n", Weighted_cordix)
    for index, col in enumerate(X.columns):
        row = Weighted_cordix[index]
        dist_corr[col] = sum(row)
    
    return dist_corr


def get_distance_corr(X, dist_corr):
    dist_corr_sum = 0
    for cols in X.columns:
        dist_corr_sum = dist_corr_sum + dist_corr[cols]
    return dist_corr_sum

if __name__ == "__main__":
    X,Y = get_data('', sample = True)
    #print(X, type(X), X.shape)
    #print(Y)
    print("X shape = ", X.shape)
    print("Y shape = ", Y.shape)

    dist_corr = distance_correlation_dict_gen(X,Y)
    print(dist_corr)
    X= X.drop(columns= ['0', '2'])
    print("subset X = ", X)
    result = get_distance_corr(X, dist_corr)
    print("result = ", result)

    












    """
    filename = ''
    #Step 1 : Get the input and label Matrix 
    X,Y = get_data(filename, sample = True)
    print(X, type(X), X.shape)
    print(Y)
    
    #Step 2 : Get Covariance Distance Matrix
    cordix_matr = get_distance_corrlation_matrix(X,Y)
    print(cordix_matr)
    

    #Step 3:  Get Label Weights
    label_weights = get_label_weights(Y)
    print("label weights = ", label_weights)
    
    #Step 4 : Get weighted cordix
    Weighted_cordix = get_weighted_cordix(cordix_matr, label_weights)  
    print("Weights_Cordix = \n", Weighted_cordix)

    #row_sum = get_row_sum(Weighted_cordix)
    row_sum = get_row_sum([[1,1,1], [10,20,30]])
    print("row sum = ", row_sum)
    #print("type of row sum = ", type(row_sum))
    
    Euclid_Dist_Mat = EDM(Weighted_cordix)
    #print("Eculidian Distance Matrix = \n", Euclid_Dist_Mat)

    Corr_distance_Vector = CDV(Euclid_Dist_Mat)
    #print("Corr_distance_Vector = \n",Corr_distance_Vector)
    """