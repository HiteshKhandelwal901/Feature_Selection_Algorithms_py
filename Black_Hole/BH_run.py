import pandas as pd 
from utility import hamming_score, hamming_scoreCV

def avg(lst):
    return sum(lst)/len(lst)

if __name__ == "__main__":
    """
    data = pd.read_csv('BH_bipirate_binary_BH_train_test.csv')

    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    X = X.iloc[:, 1:]

    print(X)
    print(Y)

    score, loss = hamming_score(X,Y)

    #score, clf, correct, incorrect = hamming_scoreCV(X, Y)
    print("score = {} loss {}".format(score, loss))
    """
    """
    data = pd.read_csv('subset_data/Flags_2features.csv')
    Y = data.iloc[:, -7:]
    X = data.iloc[:, 1: -7]
    print(X)
    print(Y)
    score, loss = hamming_score(X,Y)
    print("score = {} loss {}".format(score, loss))
    """

    """
    data = pd.read_csv('BH_continous_scene0.00051000iter.csv')
    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    X = X.iloc[:, 1:]

    print(X)
    print(Y)

    loss, rl_loss, avg_precision = hamming_score(X,Y, metric = True)
    print("hamming's loss = ", loss)
    print("rl_loss = ", rl_loss)
    print("avg_precision = ", avg_precision)
    #print("covg_error = ", covg_error)
    """

    data = pd.read_excel('reports/report_flip_yeast_20stars_0.02_lam0.0000005.xlsx')
    test_loss = data['test_loss']
    rl_loss = data['rl_loss']
    avg_precision = data['avg_precision']
    feature_size = data['feature_size']

    print("Test loss avg :", avg(test_loss))
    print("rl loss avg :", avg(rl_loss))
    print("avg precision  :", avg(avg_precision))
    print("feature size avg :", avg(feature_size))


    