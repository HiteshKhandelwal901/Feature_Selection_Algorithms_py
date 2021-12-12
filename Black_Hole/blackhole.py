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
from utility import hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations
import sklearn
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

dim = 20


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

    def updateFitness(self, X, Y):
        #print("position  = ", self.pos)
        self.fitness,self.correct,self.incorrect = self.Obj_fun(X,Y) #set this to objective function
  
    def Obj_fun(self, X, Y):
        #print("inside Pbjective Function")
        feature_index = self.select_features()
        #print("feature index = ", feature_index)
        score, correct, incorrect = self.get_score(feature_index, X, Y)
        #print("score = ", score)
        return score, correct, incorrect
    
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

    def get_score(self, feature_index, X, Y):
        size = X.shape[1]
        column_names = []
        index_to_names = defaultdict()
        #creating a dictionary of index to column names for next step
        for index,col in enumerate(X.columns):
            index_to_names[index] = col

        #print("index_to_names = ", index_to_names)
        #get the column names from the feature index that has be removed from X
        for index in feature_index:
            column_names.append(index_to_names[index])
        
        #print("column names = ", column_names)
        
        X = X.drop(columns = column_names)
        #print("X after removing features = ", X)
        
        
        if X.shape[1] > 0:
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
            term1 = (1*(ratio))
            #print("term1 = ", term1)
            term2 = feature_correlation_sum(X)
            #print("term2 = ", term2)
            #corr_X  = X.corr(method ='pearson').abs()
            #sum_corr_X = sum(X.corr)
            #term_2 = get_all_correlations(X,Y)
            term_3 = get_max_label_correlations(X,Y)
            #fitness = score- (0.6*(ratio))
            fitness = score - (0.6*ratio) - (0.5*term2) + (0.5*term_3)
            return fitness, correct, incorrect
            #return score
        else:
            #print("X is None")
            return 0,0,0

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)


    
def fit(num_of_samples,num_iter, X, Y):
    # Initializing number of stars 
    pop_number = num_of_samples
    #list to append the stars
    pop = []
    for i in range(0, pop_number):
        pop.append(Star())
        #print("Star {} pos  = {}".format(i, pop[i].pos))


    max_iter, it= num_iter, 0
    best_BH = Star()
    best_fitness = 0
    #print("intialized blackhole position = ", BH.pos, " with fitness = ", BH.fitness)

    while it < max_iter:
        print("iloop iter || ", it)
        #For each star, evaluate the objective function
        for i in range(0, pop_number):
            #print("updating fitness for star ", i)
            #each start you update its fitness value
            pop[i].updateFitness(X, Y)
            pop[i].isBH = False
            #print("Star ",i, " fitness value = ", pop[i].fitness)

        #print("done updating fitness and now finding the new blackhole\n")

        BH = pop[selectBH(pop)]
        #check if the new black hole is fitter than the previous ones
        #if it  is not then 
        #print("the best local blackhole position = ", BH.pos, " Score = ", BH.fitness)
        BH.isBH = True
        #print("comapring with global black hole")
        if BH.fitness > best_fitness:
            #print("found new global blackhole")
            best_BH_position = BH.pos
            best_fitness = BH.fitness
            best_correct  = BH.correct
            best_incorrect = BH.incorrect
            #print("global blackhole = ", best_BH_position, " fitness = ", best_fitness, "\n\n\n")
        else:
            pass
            #print("same old global blackhole = ", best_BH_position, best_fitness)
            
        #print("updating the location of the other stars")
        for i in range(pop_number):
            pop[i].updateLocation(BH)
            #print("star ", i, " new location = ", pop[i].pos)

        #print("fitness of black hole in iteration {} {}".format(BH.fitness, it))
        #print("best local black hole in iteration {} {}".format(BH.fitness, it))

        eventHorizon = calcEvetHorizon(BH, pop)
        #print("eventHorizon = ", eventHorizon)

        for i in range(pop_number):
            if isCrossingEventHorizon(BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                #print("true -crossing event horizon")
                for j in range(dim):
                    pop[i].pos[j] = pop[i].random_generator()
                #print("new random for star", i,"  = ",pop[i].pos)
        
        print("fitness || ", best_fitness, "\n")
        it = it + 1
        #break
        
    
    #
    # 
    # ("AT THE END BEST BH POSITION = ", best_BH_position)
    features = select_features_final(best_BH_position)
    print("best BH position = ", best_BH_position)
    #print("returning features = ", features)
    

    
    return features, best_fitness, best_correct, best_incorrect



if __name__ == "__main__":
    print("running driver code")

    data = pd.read_csv('Amino_MultiLabel_Dataset.csv') 
    print("info = : rows = ", data.shape[0], "column  = ", data.shape[1])
    print("data without header \n\n", data)
    column_names = []
    for i in range(data.shape[1]):
        column_names.append(str(i))
    #print("done assigning column names", column_names)
    
    data_updated = pd.read_csv('Amino_MultiLabel_Dataset.csv', names = column_names)
    #print("done adding headers \n")
    #print("data updated = \n\n", data_updated)
    #print("data columns = ", data_updated.columns)
    #print("testing columns")

    #print("data[1] = \n\n", data_updated['1'])

    Y = data_updated[['20','21','22','23']]
    print("Y = \n\n", Y)

    X = data_updated.drop(columns = Y)
    print("X = \n\n", X)

    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))

    worst_features, best_fitness, best_correct, best_incorrect = fit(5,50,X,Y)
    X= X.drop(X.columns[worst_features], axis = 1)
    print("features eliminated = ", worst_features)
    print("best subset = ", X)
    print("best fitness for these features = ", best_fitness)
    print("best correct incorrect = ", best_correct, best_incorrect)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)
    accuracy, clf, correct, incorrect = hamming_scoreCV(X_train,Y_train)
    y_pred = clf.predict(X_test).toarray()
    y_test = Y_test.to_numpy()
    score, correct, incorrect = hamming_get_accuracy(y_pred, y_test)
    print("Hamming accuracy info :\n score = {} \n incorrect prediction = {}".format(score,sklearn.metrics.hamming_loss(Y_test, y_pred)))
    print("SCORE : ", score)
    print("CORRECT : ", correct)
    print("INCORRECT : ", incorrect)



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




    
