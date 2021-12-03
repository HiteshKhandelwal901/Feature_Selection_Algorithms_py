import pandas as pd
import random
import math
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


#Step 1 :  Read the data

data = pd.read_csv('Iris.csv')
Y = data['Species']
X = data.drop(columns = ['Id', 'Species'])

#Step 2 :  Set the parametrs like dimensions 
dim = 4
range_d = 200.0

#Step 3:   Create Stars
class Star:
    def __init__(self):
        self.pos =  [self.random_generator() for i in range(dim)]
        self.isBH = False
        self.fitness = 0.0

    def updateFitness(self):
        #print("updating fitnessing value")
        self.fitness = self.Obj_fun() #set this to objective function
  
    def Obj_fun(self):
        #print("inside Pbjective Function")
        feature_index = self.select_features()
        #print("feature index = ", feature_index)
        score = self.get_score(feature_index)
        #print("score = ", score)
        return score
    
    def select_features(self):
        #print("inside feature_index")
        feature_index = []
        for index,dim in enumerate(self.pos):
            if dim<0.7:
                feature_index.append(index)
        return feature_index
    
    def get_score(self, feature_index):
        #print("insdie score")
        column_names = []
        data = pd.read_csv('Iris.csv')
        Y = data['Species']
        X = data.drop(columns= ['Species', 'Id'])

        index_to_names = {0: 'SepalLengthCm', 1: 'SepalWidthCm', 2: 'PetalLengthCm', 3: 'PetalWidthCm'}

        for index in feature_index:
            column_names.append(index_to_names[index])
        
        #print("column names = ", column_names)
        
        X = X.drop(columns = column_names)
        #print("X after removing features = ", X)

        if X.shape[1] > 0:
            #print("X is not None")
            le = preprocessing.LabelEncoder()
            Y = le.fit_transform(Y)
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

            LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, Y_train)
            score = LR.score(X_test,Y_test)
            return score
        else:
            #print("X is None")
            return 0
        





    def updateLocation(self, BH):
        #print("inside updateLocation")
        for i in range(len(self.pos)):
            rand_num = self.random_generator()
            self.pos[i] += rand_num * (BH.pos[i] - self.pos[i])


    def random_generator(self):
        num = random.uniform(0, 1)
        return num

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)

#Step 4:   Find the fittest and assign this as the black hole

def selectBH(stars):
    """Returns index of star which became black hole"""
    tmp = Star()
    tmp.fitness = -1
    it = 0
    bhNum = 0
    for star in stars:
        if star.fitness > tmp.fitness:
            tmp = star
            bhNum = it
        it += 1
    return bhNum

#Step 5:   Use the eq to chanhge all stars and push towards the blackhole

#step 6:   collapse those outside radius based on the equation
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

"""

#repeat until convergence

for numer in range(30):
    print("iteration || ", numer)

    # Initializing number of stars 
    pop_number = 3
    #list to append the stars
    pop = []
    for i in range(0, pop_number):
        pop.append(Star())

    max_iter, it= 1e8, 0
    BH = Star()
    while it < max_iter:
        #For each star, evaluate the objective function
        for i in range(0, pop_number):
            #each start you update its fitness value
            pop[i].updateFitness()
            pop[i].isBH = False
        #Select the best star that has the best fitness value as the black hole
        #this is equal to If a star reaches a location with lower cost than the black hole, exchange their locations
        BH = pop[selectBH(pop)]
        BH.isBH = True
        #Change the location of each star
        for i in range(pop_number):
            pop[i].updateLocation(BH)

        eventHorizon = calcEvetHorizon(BH, pop)

        #If a star crosses the event horizon of the black hole,
        # replace it with a new star in a random location in the search space
        for i in range(pop_number):
            if isCrossingEventHorizon(BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                for j in range(dim):
                    # ------ Problem if dimensions vary -------
                    pop[i].pos[j] = random.random() * range_d 

        #to see progress while running
        if it % 10000 == 0:
            print (BH)

        it += 1

        #breaking condition
        if BH.fitness < 1e-4:
            break

"""

if __name__ == "__main__":
    print("running the drive code")
    #s = Star()
    #print("position of star = ", s.pos)

    #s.updateFitness()

    #print("after update, fitness value = ", s.fitness)

    #s.updateLocation()

    #print("after update new pos value = ", s.pos)

    for numer in range(1):
        #print("iteration || ", numer)
        # Initializing number of stars 
        pop_number = 3
        #list to append the stars
        pop = []

        pop = []
        for i in range(0, pop_number):
            pop.append(Star())
            #print("Star {} pos  = {}".format(i, pop[i].pos))


        max_iter, it= 10, 0
        BH = Star()
        #print("intialized blackhole position = ", BH.pos, " with fitness = ", BH.fitness)

        while it < max_iter:
            print("inner while loop iter || ", it)
            #For each star, evaluate the objective function
            for i in range(0, pop_number):
                #print("updating fitness for star ", i)
                #each start you update its fitness value
                pop[i].updateFitness()
                pop[i].isBH = False
                #print("Star ",i, " fitness value = ", pop[i].fitness)

            #print("done updating fitness and now finding the new blackhole")

            BH = pop[selectBH(pop)]
            #print("the best blackhole position = ", BH.pos, " Score = ", BH.fitness)
            BH.isBH = True
                
            #print("updating the location of the other stars")
            for i in range(pop_number):
                pop[i].updateLocation(BH)
                #print("star ", i, " new location = ", pop[i].pos)
            print("fitness of black hole in iteration {} {}".format(BH.fitness, it))

            eventHorizon = calcEvetHorizon(BH, pop)

            for i in range(pop_number):
                if isCrossingEventHorizon(BH, pop[i], eventHorizon) == True and pop[i].isBH == False:
                    for j in range(dim):
                        pop[i].pos[j] = pop[i].random_generator()
            
            
            it = it + 1
        









