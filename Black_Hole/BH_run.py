import pandas as pd 
from utility import hamming_scoreCV



if __name__ == "__main__":
    data = pd.read_csv('subset_data/BH_bipirate.csv')

    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    X = X.iloc[:, 1:]

    print(X)
    print(Y)

    score, clf, correct, incorrect = hamming_scoreCV(X, Y)