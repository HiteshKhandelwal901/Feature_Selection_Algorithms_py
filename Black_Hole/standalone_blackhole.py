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
from utility import weighted_label_correlations_70_30,hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations, get_index_sum, get_max_label_correlations_gen, get_max_corr_label
import sklearn
from filters import remove_features, univariate_feature_elimination
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

dim = 279
score_cache = defaultdict()



def get_index_sum(X, cols):
    index_sum = []
    for col in cols:
        index_sum.append(X.columns.get_loc(col))
        return sum(index_sum)

def selectBH(stars):
    """Returns index of star which became black hole"""
    tmp = Star("temp")
    tmp.fitness = 0
    it = 0
    bhNum = 0
    for star in stars:
        if star.isBH == False:
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


def select_worst_features(pos):
    feature_index = []
    for index,dim in enumerate(pos):
        if dim<0.5:
            feature_index.append(index)
    return feature_index




class Star:
    def __init__(self,name):
        self.pos =  [self.random_generator() for i in range(dim)]
        self.isBH = False
        self.fitness = 0.0
        self.correct = 0
        self.incorrect = 0
        self.ham_loss = 0
        self.ham_score = 0
        self.name = name

    def updateFitness(self,label_dict, X, Y):
        self.fitness, self.ham_score, self.ham_loss = self.Obj_fun(label_dict, X,Y) #set this to objective function
  
    def Obj_fun(self, label_dict, X, Y):
        feature_index = self.select_features()
        score, ham_score, ham_loss = self.get_score(label_dict,feature_index, X, Y)
        return score, ham_score, ham_loss
    
    def select_features(self):

        feature_index = []
        for index,dim in enumerate(self.pos):
            if dim<0.5:
                feature_index.append(index)
        return feature_index


    def updateLocation(self, BH):
        for i in range(len(self.pos)):
            rand_num = self.random_generator()
            self.pos[i] += rand_num * (BH.pos[i] - self.pos[i])
        
    def random_generator(self):
        num = random.uniform(0, 1)
        return num

    def get_score(self, label_dict,feature_index, X, Y):
        #print("feature_index = ", feature_index)
        size = X.shape[1]
        column_names = []
        index_to_names = defaultdict()

        #creating a dictionary of index to column names for next step
        
        for index,col in enumerate(X.columns):
            index_to_names[index] = col
        
        #get the column names from the feature index that has be removed from X
        for index in feature_index:
            column_names.append(index_to_names[index])
        

        X = X.drop(columns = column_names, axis = 1)
        index_sum = sum(feature_index)

        
        
        
        if X.shape[1] > 0:

            if index_sum in score_cache:
                fitness, ham_score, ham_loss = score_cache[index_sum]
                return fitness, ham_score, ham_loss
        

            score,clf,correct, incorrect = hamming_scoreCV(X, Y)
            features_selected = (size - len(feature_index))
            fitness = score / (1 + (0.2*features_selected))
            score_cache[index_sum] = (fitness, score,1-score)
            return fitness, score, (1-score)
        else:
            return 0,0,0

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)


    
def fit(num_of_samples,num_iter, X, Y):
    """
    function to run blackhole feature selection algorithm

    Args:

    num_of_samples : population size
    num_iter :  max_iterations
    X :  Dataframe of attributes. Headers is required
    Y :  Dataframe of labels. Headers is required

    returns:

    Ham score : Hamming's score
    Ham loss :  Hamming's loss
    Worst features :  Blackhole's feature index whose value is lesser than threshold (0.5). Just remove these features
    
    
    """





    # Initializing number of stars 
    pop_number = num_of_samples
    #list to append the stars
    pop = []
    for i in range(0, pop_number):
        pop.append(Star(str(i)))



    max_iter, it= num_iter, 0
    global_BH = Star(name = "default")
    global_BH.isBH = True
    global_BH.fitness = 0

  
    label_dict = weighted_label_correlations_70_30(X,Y)
    while it < max_iter:
        print("iloop iter || ", it)
        for i in range(0, pop_number):
            if pop[i].isBH == False:
                pop[i].updateFitness(label_dict,X, Y)
            else:
                pass


        local_BH = pop[selectBH(pop)]
        if local_BH.fitness > global_BH.fitness:
            global_BH.isBH = False
            global_BH = local_BH
            global_BH.isBH = True
        else:
            pass
          

        for i in range(pop_number):
            if pop[i].isBH == False:
                pop[i].updateLocation(global_BH)
               
            else:
                pass
                
        eventHorizon = calcEvetHorizon(global_BH, pop)

        for i in range(pop_number):
            if isCrossingEventHorizon(global_BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                for j in range(dim):
                    pop[i].pos[j] = pop[i].random_generator()

        print("fitness || ", global_BH.fitness, "\n")
        features = select_worst_features(global_BH.pos)
        print("hamming's loss = ", global_BH.ham_loss)
        print("ham score = ", global_BH.ham_score)
        print("number of features selected = ", (dim-len(features)))

        print("\n\n")
        it = it + 1
    
    #Done training
    worst_features = select_worst_features(global_BH.pos)
    X_final= X.drop(X.columns[worst_features], axis = 1)

    

    print("---END OF ALGORITHM-----\\n\n")
    print("best subset size = ", X_final.shape)
    print("hamming's loss = ",global_BH.ham_loss)
    print("hamming's score = ", global_BH.ham_score)
    print("Done saving the best subset as csv file \n\n")
    df = pd.concat((X_final, Y), axis = 1)
    df.to_csv('best_subset.csv')
    return X_final, global_BH.ham_score, global_BH.ham_loss



if __name__ == "__main__":

    #Reading the data into Dataframe
    data = pd.read_csv("scene.csv")

    #Get X and Y from the data
    Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
    X = data.drop(columns= Y)
    
    #uncomment to run with chi^2
    #X = univariate_feature_elimination(X,Y,15)

    print("running without ch^2")
    #print the information about X and Y
    print("INFO : \n\n")
    print("X shape : ", X.shape)
    print("X type = ", type(X))
    print("Y shape = : ", Y.shape)
    print("Y type: ", type(Y))
 
    #Run without BH, just the random forest CV
    print("\n\n-----without feature selection ----- \n\n")
    
    #Get trainCV score and subract it from 1 to get loss
    CVscore, clf, correct, incorrect = hamming_scoreCV(X,Y)
    print("trainCV hamming's loss :", 1-CVscore)
    
    #Run with BH
    print("\n\n---with feature selection lambda = 0.4------\n\n")
    
    #Get the fitness, ham score, ham loss and the worst features
    X_subset , ham_score, ham_loss = fit(20,50,X,Y)