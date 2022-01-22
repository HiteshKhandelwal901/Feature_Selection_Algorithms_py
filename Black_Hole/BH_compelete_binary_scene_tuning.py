from collections import defaultdict
import warnings,os
import pandas as pd
import random
import math
import sys
from scipy.stats.morestats import Variance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_multilabel_classification
from utility import hamming_score, weighted_label_correlations_70_30,hamming_scoreCV, hamming_get_accuracy, feature_correlation_sum, get_max_label_correlations, get_index_sum, get_max_label_correlations_gen, get_max_corr_label
import sklearn
from filters import remove_features, univariate_feature_elimination
from Bipirate_Algorithm import distance_correlation_dict_gen, euclid_dist, get_distance_corr
import numpy as np
from sklearn.metrics import hamming_loss
import copy
from sklearn.metrics import label_ranking_loss
import statistics 
import time
from sklearn.metrics import label_ranking_loss, label_ranking_average_precision_score, coverage_error
from skmultilearn.adapt import MLkNN


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

dim = 4
score_cache = defaultdict()



def create_subset(df, dst_df):
    cols = df.columns
    final_df = dst_df[cols]
    return final_df




def get_index_sum(X, cols):
    index_sum = []
    for col in cols:
        index_sum.append(X.columns.get_loc(col))
        return sum(index_sum)

def selectBH(stars):
    """Returns index of star which became black hole"""
    tmp = Star("temp")
    tmp.fitness = -10000000
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
        #print("r = ", r)
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
        self.pos =  [self.random_generator_binary() for i in range(dim)]
        self.isBH = False
        self.fitness = -1000000000
        self.correct = 0
        self.incorrect = 0
        self.ham_loss = 1
        self.ham_score = 0
        self.name = name
        self.clf = MLkNN(k=10)
    
    def random_generator_binary(self):
        num = random.uniform(0, 1)
        if num > 0.5:
            return 1
        else:
            return 0


    def updateFitness(self,lam, label_dict, X, Y):
        self.fitness, self.ham_score, self.ham_loss, self.clf = self.Obj_fun(lam, label_dict, X,Y) #set this to objective function
  
    def Obj_fun(self,lam, label_dict, X, Y):
        feature_index = self.select_features()
        fitness, ham_score, ham_loss, clf = self.get_score(lam, label_dict,feature_index, X, Y)
        return fitness, ham_score, ham_loss, clf
    
    def select_features(self):

        feature_index = []
        for index,dim in enumerate(self.pos):
            if dim<0.5:
                feature_index.append(index)
        return feature_index

    def switch_activation(self):
        #loop through each dimension
        for i in range(len(self.pos)):
            #print("dim {}".format(i))
            rand_num = random.uniform(0, 1)
            #print("rand_num = ", rand_num)
            if rand_num <= 0.5:
                #print("true lesser than 0.05")
                #convert 0 to 1
                if self.pos[i] == 0:
                    #print("changing zero to 1")
                    self.pos[i] = 1
                #convert 1 to 0
                elif self.pos[i] == 1:
                    #print("changing 1 to 0")
                    self.pos[i] = 0
            else:
                pass
        #print("exiting after activation with new pos = ", self.pos)


    def random_generator_vector(self):
        return [random.uniform(0, 1) for i in range(dim)]

    def updateLocation_binary(self, BH):
        for i in range(len(self.pos)):
            if self.pos[i] == BH.pos[i]:
                pass
            else:
                rand_num = self.random_generator()
                if rand_num < 0.5:
                    self.pos[i] = copy.copy(BH.pos[i])
    
    def updateLocation(self, BH):
        for i in range(len(self.pos)):
            rand_num = self.random_generator()
            self.pos[i] += rand_num * (BH.pos[i]-self.pos[i])
        
    def random_generator(self):
        num = random.uniform(0, 1)
        return num

    def get_score(self,lam,label_dict,feature_index, X, Y):
        """
        Function to get the fitness score 

        Args:
        X,Y : attr and labels in DF with headers req
        feature_index = list of features to drop from X for this star
        label_dict = dicitonary of dist_corr for each attribute

        returns fitness, score, loss
        """

        size = X.shape[1]
        column_names = []
        index_to_names = defaultdict()

        #creating a dictionary of index to column names for next step
        for index,col in enumerate(X.columns):
            index_to_names[index] = col
        
        #get the column names from the feature index that has be removed from X
        for index in feature_index:
            column_names.append(index_to_names[index])
        
        #get the subset for this satr
        X = X.drop(columns = column_names, axis = 1)
        index_sum = sum(feature_index)

        
        
        #if subset is not empty
        if X.shape[1] > 0:
            #if the subset is alreasy seen before, get the score and loss from cache
            if index_sum in score_cache:
                fitness, ham_score, ham_loss,clf = score_cache[index_sum]
                return fitness, ham_score, ham_loss,clf
        
            #if the subset is not seen before, get the score by running hamming CV
            #score,clf,correct, incorrect = hamming_scoreCV(X, Y)
            score, loss, clf = hamming_score(X,Y)
            #score, loss = self.hamming_score(X,Y)

            #Num of features selected
            features_selected = (size - len(feature_index))
            corr_dist_sum = get_distance_corr(X,label_dict)
            #fitness equation
            fitness = (score / (1 + (lam*features_selected))) - (0.5*corr_dist_sum)
            #cache the information for this subset. cache based on feature_index, i.e, sum of index of features to remove
            score_cache[index_sum] = (fitness, score,1-score, clf)
            #print("going to return")
            return (fitness, score, loss , clf)
        else:
            #print("inside else")
            return 0,0,0,0

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)



def binary_pos(pos):
    binary_list= []
    for p in pos:
        if p >= 0.5:
            binary_list.append(1)
        else:
            binary_list.append(0) 
    return binary_list

def fit(lam, num_of_samples,num_iter, X, Y):
    """
    function to run blackhole feature selection algorithm

    Args:

    lam : parameter
    num_of_samples : population size
    num_iter :  max_iterations
    X :  Dataframe of attributes. Headers is required
    Y :  Dataframe of labels. Headers is required

    returns:

    Ham score : Hamming's score
    Ham loss :  Hamming's loss
    Best subset
    """
    # Initializing number of stars 
    pop_number = num_of_samples
    #list to append the stars
    pop = []
    for i in range(0, pop_number):
        pop.append(Star(str(i)))
        #print("start {} initalized pos {}".format(pop[i].name, pop[i].pos))


    #intialize parametrs and a global black hole ( best black hole)
    max_iter, it= num_iter, 0
    global_BH = Star(name = "default")
    #print("default global BH pos = ", global_BH.pos)
    global_BH.isBH = True

    #get the bipirate distance_correlation dictionary for all ( X,Y)
    label_dict = distance_correlation_dict_gen(X,Y)
    #print("lable_dict = \n", label_dict)

    #start the loop
    while it < max_iter:
        print("iloop iter || ", it)

        #intialize the population of stars and update thier fitnes
        for i in range(0, pop_number):
            if pop[i].isBH == False:
                pop[i].updateFitness(lam, label_dict,X, Y)
            else:
                pass

        #get the best star in this current iteration
        local_BH = pop[selectBH(pop)]

        #if the best star in this iteration is fitter than global make that global
        if local_BH.fitness > global_BH.fitness:
            global_BH.isBH = False
            global_BH = local_BH
            global_BH.isBH = True
        else:
            pass

        #update the location of other stars
        #print("updating location of stars")
        for i in range(pop_number):
            if pop[i].isBH == False:
                pop[i].updateLocation_binary(global_BH)
                #print("star {} after location update {}".format(i, pop[i].pos))
            if it%5 == 0 and it>0:
                #print("it % 5 true")
                #print("before activation, start {} pos = {} ".format(i, pop[i].pos))
                pop[i].switch_activation()
                #print("after activation, start {} pos = {} ".format(i, pop[i].pos))

        #get the event horizon        
        eventHorizon = calcEvetHorizon(global_BH, pop)
        

        #if in event horizon, reintialize stars randomly
        for i in range(pop_number):
            if isCrossingEventHorizon(global_BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                #print("start {} crossing event horizion".format(i))
                for j in range(dim):
                    #pop[i].pos[j] = pop[i].random_generator()
                            num = random.uniform(0, 1)
                            if num > 0.5:
                                pop[i].pos[j] = 1
                            else:
                                pop[i].pos[j] = 0
                #print("newly intialized position of star {} is {}".format(i,pop[i].pos))
                    

        #print("fitness || ", global_BH.fitness, "\n")
        #print("lam = ", lam)
        features = select_worst_features(global_BH.pos)
        #print("hamming's loss = ", global_BH.ham_loss)
        #print("ham score = ", global_BH.ham_score)
        #print("number of features selected = ", (dim-len(features)))

        #print("\n\n")
        #print("converting BH to binary")

        it = it + 1
    
    #print("sample star pos = ", pop[12].pos)
    #print("blackhole pos = ", global_BH.pos)
    #Done training
    worst_features = select_worst_features(global_BH.pos)
    X_final= X.drop(X.columns[worst_features], axis = 1)

    
    

    #print("---END OF ALGORITHM-----\\n\n")
    #print("best subset size = ", X_final.shape)
    #print("hamming's loss = ",global_BH.ham_loss)
    #print("hamming's score = ", global_BH.ham_score)
    #print("Done saving the best subset as csv file \n\n")
    #df = pd.concat((X_final, Y), axis = 1)
    #name = 'BH_complete_binary_yeast' + str(lam) + '.csv'
    #print("saving {} ".format(name))
    #df.to_csv(name)
    return X_final,worst_features, global_BH.ham_score, global_BH.ham_loss, global_BH.clf

def Average(lst):
    return sum(lst) / len(lst)

def variance(lst):
    return statistics.variance(lst)


if __name__ == "__main__":

   
    #print("X_train shape", X_train.shape)
    #print("X_test shape", X_test.shape)
    #print("Y_train shape", Y_train.shape)
    #print("Y_test shape = ", Y_test.shape)
 
    
    #parameters and variables intializations
    lam = 0.0005
    loss_list = []
    feature_list = []
    rl_loss_list = []
    avg_precision_list = []

    #number of runs of program, right now only 1 run.
    start_time = time.time()
    for runs in range(1):
        print("---RUN {}---".format(runs))
        random.seed(runs)
        seed = random.randint(1, 1000)
        #Reading the data into Dataframe
        data = pd.read_csv("Data/scene.csv")

        #Get X and Y from the data
        Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
        X = data.drop(columns= Y)
        #X = X.iloc[:, 1:10]
        

        scaled_features = sklearn.preprocessing.MinMaxScaler().fit_transform(X.values)
        X = pd.DataFrame(scaled_features, index= X.index, columns= X.columns)
        #uncomment to run with chi^2
        X = univariate_feature_elimination(X,Y,15)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=seed)
        print("X train shape = ", X_train.shape)
        print("X test shape = ", X_test.shape)
        print("Y train shape", Y_train.shape)
        print("Y test shape = ", Y_test.shape)
        #run the algorithm
        X_subset , inactive_features, train_score, train_loss, clf = fit(lam,5,10,X_train,Y_train)
        #print("inactive features \n\n", inactive_features)
        #print("length of inactive features = ", len(inactive_features))
        X_test_subset = X_test.drop(X_test.columns[inactive_features],axis=1)
        #print("X train subset = ", X_subset.shape)
        #print("X_test ater drop", X_test_subset.shape)
        #print("X_subset \n", X_subset.columns)
        #print("X_test \n",X_test_subset.columns)
        #print(clf)
        #print("X_test shape = ", X_test.shape)
        #print("best clf = ", clf)
        #clf = MLkNN(k=10)
        #caculate rl_loss, avg_prec for the best subset
        #test_loss, rl_loss, avg_precision = hamming_score(X_test,Y_test, metric = True)
        y_pred = clf.predict(X_test_subset).toarray()
        test_loss = hamming_loss(y_pred, Y_test)
        rl_loss = label_ranking_loss(Y_test,y_pred)
        avg_precision = label_ranking_average_precision_score(Y_test, y_pred)
        #append all metrics to list to calculate avg 
        loss_list.append(test_loss)
        rl_loss_list.append(rl_loss)
        avg_precision_list.append(avg_precision)
        feature_list.append(X_subset.shape[1])
        print("train loss {} features {}".format(train_loss, X_subset.shape[1]))
        print("test loss with BH = {} and features selected = {}".format(test_loss, X_test_subset.shape[1]))
        print("rl loss || prrcision {} {} ".format(rl_loss, avg_precision))
    print("--- %s seconds ---" % (time.time() - start_time))
    print("losst list \n\n {}".format(loss_list))
    print("rl loss list \n\n{}".format(rl_loss_list))
    print("precion list \n\n {}".format(avg_precision_list))
    print("features list \n\n{}".format(feature_list))
    print("avg ham loss = ", Average(loss_list))
    print("avg rl loss = ", Average(rl_loss_list))
    print("avg of avg precision = ", Average(avg_precision_list))
    print("AVG features selected = ", Average(feature_list))
    print("variance of ham loss {}".format(variance(loss_list)))
    print("variance of rl loss {}".format(variance(rl_loss_list)))
    print("variance of presicion loss {}".format(variance(avg_precision_list)))
    print("variance of features size {}".format(variance(feature_list)))

    

    