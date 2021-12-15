"""
V5 -- dipitide data with only 3rd term and score cash
"""


"""
Score cache with one term on Amino data and di
"""

from collections import defaultdict
import warnings,os
import pandas as pd
import random
import math
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_multilabel_classification
from utility import hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations, get_index_sum, get_max_label_correlations_gen, get_max_corr_label
import sklearn
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

dim = 6
score_cache = defaultdict()



def get_index_sum(X, cols):
    index_sum = []
    for col in cols:
        index_sum.append(X.columns.get_loc(col))
        return sum(index_sum)

def selectBH(stars):
    """Returns index of star which became black hole"""
    tmp = Star()
    tmp.fitness = 0
    it = 0
    bhNum = 0
    for star in stars:
        if star.fitness > tmp.fitness:
            tmp = star
            bhNum = it
        it += 1
    return bhNum


def calcEvetHorizon(BH, stars):
    tmp = 0
    for star in stars:
        tmp += star.fitness
    return BH.fitness / tmp

def isCrossingEventHorizon(BH, star, horizon):
    r = 0.0
    #euclidian norm
    for i in range(len(star.pos)):
        r += pow(star.pos[i] - BH.pos[i], 2)
    if math.sqrt(r) <= horizon:
        return True
    return False


def select_features_final(pos):
    #print("inside feature_index")
    feature_index = []
    for index,dim in enumerate(pos):
        if dim<0.5:
            feature_index.append(index)
    return feature_index




class Star:
    def __init__(self):
        self.pos =  [self.random_generator() for i in range(dim)]
        self.isBH = False
        self.fitness = 0.0
        self.correct = 0
        self.incorrect = 0
        self.ham_loss = 0
        self.ham_score = 0
        self.name = 0

    def updateFitness(self,label_dict,constant1, X, Y):
        #print("position  = ", self.pos)
        self.fitness, self.ham_score, self.ham_loss = self.Obj_fun(label_dict,constant1, X,Y) #set this to objective function
  
    def Obj_fun(self, label_dict,constant1, X, Y):
        #print("inside Pbjective Function")
        feature_index = self.select_features()
        #print("feature index = ", feature_index)
        score, ham_score, ham_loss = self.get_score(label_dict,constant1,feature_index, X, Y)
        #print("score = ", score)
        return score, ham_score, ham_loss
    
    def select_features(self):
        #print("inside feature_index")
        feature_index = []
        for index,dim in enumerate(self.pos):
            if dim<0.5:
                feature_index.append(index)
        return feature_index


    def updateLocation(self, BH):
        #print("inside updateLocation")
        for i in range(len(self.pos)):
            rand_num = self.random_generator()
            self.pos[i] += rand_num * (BH.pos[i] - self.pos[i])
        
    def random_generator(self):
        num = random.uniform(0, 1)
        return num

    def get_score(self, label_dict, constant1, feature_index, X, Y):
        #print("feature_index = ", feature_index)
        size = X.shape[1]
        column_names = []
        index_to_names = defaultdict()

        #creating a dictionary of index to column names for next step

        #print("index_to_names = ", index_to_names)
        #get the column names from the feature index that has be removed from X
        for index,col in enumerate(X.columns):
            index_to_names[index] = col
        
        for index in feature_index:
            column_names.append(index_to_names[index])
        
        #print("column names = ", column_names)

        X = X.drop(columns = column_names, axis = 1)
        #print("X after removing features = ", X)
        index_sum = sum(feature_index)

        
        
        
        if X.shape[1] > 0:

            if index_sum in score_cache:
                #print("Already calculated")
                fitness, ham_score, ham_loss = score_cache[index_sum]
                #print("fitness = ", fitness)
                #print("\n\n")
                return fitness, ham_score, ham_loss
            
            #le = preprocessing.LabelEncoder()
            #Y = le.fit_transform(Y)
            #X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

            #LR = LogisticRegressionCV(max_iter = 1000, verbose = 0).fit(X_train, Y_train)
            #score = LR.score(X_test,Y_test)
            #print("length of feature index = ", len(feature_index))
            #print("X[1] shape = ", size)

            score,clf,correct, incorrect = hamming_scoreCV(X, Y)
            features_selected = (size - len(feature_index))
            #print("len of selected features = ", features_selected)
            ratio = features_selected / size
            #print("ratio = ", ratio)
            #term1 = (1*(ratio))
            #print("term1 = ", term1)
            #term2 = feature_correlation_sum(X)
            #print("term2 = ", term2)
            #corr_X  = X.corr(method ='pearson').abs()
            #sum_corr_X = sum(X.corr)
            #term_2 = get_all_correlations(X,Y)
            #term_3 = get_max_label_correlations(X,Y)
            term_3_eff = get_max_corr_label(X, label_dict)
            #print("term3 = ", term_3_eff)
            fitness = score + 0.01*term_3_eff
            #fitness = score - (constant1*term1)
            #fitness = score - term1 - (0.5*term2) + (0.5*term_3)
            score_cache[index_sum] = (fitness, score,1-score)
            return fitness, score, (1-score)
            #return score
        else:
            #print("X is None")
            #print("Empty DF")
            #print("fitness = ", 0)
            return 0,0,0

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)


    
def fit(constant1,num_of_samples,num_iter, X, Y):
    # Initializing number of stars 
    pop_number = num_of_samples
    #list to append the stars
    pop = []
    for i in range(0, pop_number):
        pop.append(Star())
        #print("Star {} pos  = {}".format(i, pop[i].pos))


    max_iter, it= num_iter, 0
    best_BH = Star()
    best_BH_position = best_BH.pos
    best_fitness = 0
    label_dict = get_max_label_correlations_gen(X,Y)
    #print("label_dict = ", label_dict)
    #print("intialized blackhole position = ", BH.pos, " with fitness = ", BH.fitness)

    while it < max_iter:
        print("iloop iter || ", it)
        #For each star, evaluate the objective function
        for i in range(0, pop_number):
            print("\n\nupdating fitness for star ", i)
            #each start you update its fitness value
            pop[i].updateFitness(label_dict,constant1,X, Y)
            pop[i].isBH = False
            pop[i].name = i
            print("star", i, "pos = ", pop[i].pos)
            print("Star ",i, " fitness value = \n", pop[i].fitness)

        #print("done updating fitness and now finding the new blackhole\n")

        BH = pop[selectBH(pop)]
        #check if the new black hole is fitter than the previous ones
        #if it  is not then 
        #print("the best local blackhole position = ", BH.pos, " Score = ", BH.fitness)
        print("in iteration {} best start = star {}".format(it, BH.name))
        print("\n\nin iter {} best star fitness is {}".format(it, BH.fitness))
        print("n\nin iter {} best star position is {}\n".format(it, BH.pos))
        BH.isBH = True
        print("best fitness so far = {}".format(best_fitness))
        print("best star position so far  = {}\n".format(best_BH_position))
        print("comapring {} > {}".format(BH.fitness , best_fitness))
        if BH.fitness > best_fitness:
            print("found new global blackhole")
            best_BH_position = BH.pos
            best_fitness = BH.fitness
            ham_loss = BH.ham_loss
            ham_score = BH.ham_score
            #print("global blackhole = ", best_BH_position, " fitness = ", best_fitness, "\n\n\n")
        else:
            pass
            print("same old global blackhole = ", best_BH_position, best_fitness)
            
        #print("updating the location of the other stars")
        for i in range(pop_number):
            pop[i].updateLocation(BH)
            #print("star ", i, " new location = ", pop[i].pos)


        eventHorizon = calcEvetHorizon(BH, pop)
        #print("eventHorizon = ", eventHorizon)

        for i in range(pop_number):
            if isCrossingEventHorizon(BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                #print("true -crossing event horizon")
                for j in range(dim):
                    pop[i].pos[j] = pop[i].random_generator()
                print("new position for star", i,"  = ",pop[i].pos)
        #print("constant value = ", constant1)
        print("best BH position = \n", best_BH_position)
        print("fitness || ", best_fitness, "\n")
        features = select_features_final(best_BH_position)
        print("hamming's loss = ", ham_loss)
        print("ham score = ", ham_score)
        print("number of features selected = ", (294-len(features)))
        #print("features eliminated = ", features)
        print("\n\n")
        it = it + 1
        #break
        
    
    #
    # 
    # ("AT THE END BEST BH POSITION = ", best_BH_position)
    features = select_features_final(best_BH_position)
    #print("best BH position = ", best_BH_position)
    #print("returning features = ", features)
    

    
    return features, best_fitness,ham_score, ham_loss



if __name__ == "__main__":

    """
    data = pd.read_csv("scene.csv")
    print("data = \n", data)
    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    print("X = \n\n", X)

    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))

    print("\n\n-----without feature selection ----- \n\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming score info for without feature selection :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
    
    print("\n\n---with feature selection------\n\n")
    worst_features, best_fitness, ham_score, ham_loss = fit(0.01, 5,10,X,Y)
    X_final= X.drop(X.columns[worst_features], axis = 1)
    
    X_final= X.drop(X.columns[worst_features], axis = 1)
    print("features eliminated = ", worst_features)
    print("best fitness for these features = ", best_fitness)

    X_train, X_test, Y_train, Y_test = train_test_split(X_final,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming accuracy info Random Forest:\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
    print("best subset = ", X_final)
    """





    """
    print("-----BENCHING ON DIPEPTIDE DATASET-------")
    data = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv')
    #print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
    #print("data without header \n\n", data)
    column_names = []
    for i in range(data.shape[1]):
        column_names.append(str(i))
    #print("done assigning column names", column_names)
    data_updated = pd.read_csv('Dipeptide_MultiLabel_Dataset.csv', names = column_names)
    Y = data_updated[['400','401','402','403']]
    print("Y = \n\n", Y)

    X = data_updated.drop(columns = Y)
    print("X = \n\n", X)

    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))

    print("\n\n-----without feature selection ----- \n\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming score info for without feature selection :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
    
    print("---with feature selection------")
    worst_features, best_fitness, ham_score, ham_loss = fit(0.01, 10,25,X,Y)
    X_final= X.drop(X.columns[worst_features], axis = 1)
    
    X_final= X.drop(X.columns[worst_features], axis = 1)
    print("features eliminated = ", worst_features)
    print("best fitness for these features = ", best_fitness)

    X_train, X_test, Y_train, Y_test = train_test_split(X_final,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming accuracy info Random Forest:\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
    print("best subset = ", X_final)
    """

    
    print("running driver code")

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

    print("\n\n-----without feature selection ----- \n\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, ham_score, ham_loss = hamming_get_accuracy(y_pred, y_test)
    print("Hamming score info for without feature selection :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )

    print("\n\n-----with feature selection ----- \n\n")
    #constants range 
    worst_features, best_fitness, ham_score, ham_loss = fit(0.01, 5,10,X_train,Y_train)
    global_max = 0
    #worst_features, best_fitness, best_correct, best_incorrect = fit(5,10,X,Y)
    X_final= X.drop(X.columns[worst_features], axis = 1)
    #print("features eliminated = ", worst_features)
    #print("best fitness for these features = ", best_fitness)
    
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_final,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming accuracy info Random Forest:\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)
    print("hamming's loss  = ",sklearn.metrics.hamming_loss(Y_test, y_pred) )
    print("best subset = ", X_final)
    



    """
    X, y = make_multilabel_classification(n_samples = 300,n_features = 20,sparse = True, n_labels = 5,
    return_indicator = 'sparse', allow_unlabeled = False)
    X = pd.DataFrame(X.toarray())
    y = y.toarray()
    print("X SHAPE  = ", X.shape,"type = ", type(X))
    print("Y shape  = ", y.shape)
    print("cols = ", X.columns)

    worst_features, best_fitness = fit(10,20,X,y)
    X= X.drop(X.columns[worst_features], axis = 1)
    print("features eliminated = ", worst_features)
    print("best subset = ", X)
    print("best fitness for these features = ", best_fitness)
    """





    """
  #---------- IRIS DATASET RESULTS ----------
    column_names = []
    data = pd.read_csv('Iris.csv')
    Y = data['Species']
    X = data.drop(columns= ['Species', 'Id'])

    #print("X after removing features = ", X)
    print("features information : ", X.shape)
    print("labels information : ", Y.shape)
    best_features, best_fitness = fit(3,3,X,Y)
    #print("best_star values = ", best_star.pos, "accuracy = ", best_star.fitness)
    #print("best subset = ", best_star.select_features())
    X= X.drop(X.columns[best_features], axis = 1)
    print("features eliminated = ", best_features)
    print("best subset = ", X)
    print("best fitness for these features = ", best_fitness)
    """


    
    """"
    #---------BREAST-CANCER DATASET BENCHMARKING--------
    Column_names = []
    index_to_names = defaultdict()
    data = pd.read_csv('breast_data.csv')
    Y = data['diagnosis']
    X = data.drop(columns = ['id', 'diagnosis'])

    for index,col in enumerate(X.columns):
        index_to_names[index] = col
    
    print("index_to_names = ",index_to_names)

    #print("X after removing features = ", X)
    print("features information : ", X.shape)
    print("labels information : ", Y.shape)
    #best_star = fit(10,20,X,Y)
    best_features, best_fitness = fit(6,20,X,Y)
    X= X.drop(X.columns[best_features], axis = 1)
    print("features eliminated = ", best_features)
    print("best subset = ", X)
    print("best fitness for these features = ", best_fitness)
    print("reduced feature space = ", X.shape)
    #print("best_star values = ", best_star.pos, "accuracy = ", best_star.fitness)
    #print("best subset = ", best_star.select_features())
    #feature_index = best_star.select_features()
    #print("number of reduced features =", feature_index)

    #cols = []

    #for each_feat_index in feature_index:
    #    cols.append(index_to_names[each_feat_index])

    #print("the best subset of feature space is  :", cols)
    #print("the corresponding accuracy is :", best_star.fitness )
    """

    """
    ----- ZOO DATASET BENCHAMRKING ----------

    Column_names = []
    index_to_names = defaultdict()
    data = pd.read_csv('zoo.csv')
    Y = data['class_type']
    X = data.drop(columns = ['animal_name', 'class_type'])

    for index,col in enumerate(X.columns):
        index_to_names[index] = col
    
    print("index_to_names = ",index_to_names)

    #print("X after removing features = ", X)
    print("features information : ", X.shape)
    print("labels information : ", Y.shape)
    best_star = fit(10,100,X,Y)
    print("best_star values = ", best_star.pos, "accuracy = ", best_star.fitness)
    print("best subset = ", best_star.select_features())
    feature_index = best_star.select_features()
    print("number of reduced features =", feature_index)

    cols = []

    for each_feat_index in feature_index:
        cols.append(index_to_names[each_feat_index])

    print("the best subset of feature space is  :", cols)
    print("the corresponding accuracy is :", best_star.fitness )
    """


    """
    ------- leukemia dataset ------
    Column_names = []
    index_to_names = defaultdict()
    data = pd.read_csv('leukemia.csv')
    Y = data[data.columns[-1]]
    print("y = ", Y)

    positive = 0
    negitive = 0
    for i,label in enumerate(Y):
        if Y[i] == "ALL":
            Y[i] = 1
            positive = positive +1
        else:
            negitive = negitive +1
            Y[i] = 0
    
    print("positive = ", positive)
    print("negitive = ", negitive)
    
    print("y = ", Y)
    X = data.iloc[:, :-1]

    for index,col in enumerate(X.columns):
        index_to_names[index] = col
    
    #print("index_to_names = ",index_to_names)

    #print("X after removing features = ", X)
    print("features information : ", X.shape)
    print("labels information : ", Y.shape)
    
    
    
    
    best_star = fit(5,20,X,Y)
    print("best_star values = ", best_star.pos, "accuracy = ", best_star.fitness)
    print("best subset = ", best_star.select_features())
    feature_index = best_star.select_features()
    print("number of reduced features =", len(feature_index))

    cols = []

    for each_feat_index in feature_index:
        cols.append(index_to_names[each_feat_index])

    print("the best subset of feature space is  :", cols)
    print("the corresponding accuracy is :", best_star.fitness )

    """




    
