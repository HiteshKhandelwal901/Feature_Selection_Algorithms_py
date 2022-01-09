import pandas as pd 
from utility import hamming_score, hamming_scoreCV



if __name__ == "__main__":
    data = pd.read_csv('BH_bipirate_binary_BH_train_test.csv')

    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    X = X.iloc[:, 1:]

    print(X)
    print(Y)

    score, loss = hamming_score(X,Y)

    #score, clf, correct, incorrect = hamming_scoreCV(X, Y)
    print("score = {} loss {}".format(score, loss))