import pandas as pd
import random
import math
import sys


#Step 1 :  Read the data

data = pd.read_csv('Iris.csv')
Y = data['Species']
X = data.drop(columns = ['Id', 'Species'])

#Step 2 :  Set the parametrs like dimensions 
dim = 3
range_d = 200.0

#Step 3:   Create Stars
class Star:
    def __init__(self):
        self.pos = None #selecting based on uniform varibale, static or varibale ?
        self.isBH = False
        self.fitness = 0.0

    def updateFitness(self):
        self.fitness = None #set this to objective function
  

    def updateLocation(self, BH):
        for i in range(len(self.pos)):
            # ---- If dimensions vary then here's the porblem ------
            self.pos[i] += random.random() * (BH.pos[i] - self.pos[i])

    def __str__(self):
        print(self.pos)
        return "Is Bh: " + str(self.isBH) + " fitness: " + str(self.fitness)

#Step 4:   Find the fittest and assign this as the black hole

def selectBH(stars):
    """Returns index of star which became black hole"""
    tmp = Star()
    tmp.fitness = sys.maxint
    it = 0
    bhNum = 0
    for star in stars:
        if star.fitness < tmp.fitness:
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
        # ---- problem if  dimensions are variable -------
        r += pow(star.pos[i] - BH.pos[i], 2)
    if math.sqrt(r) <= horizon:
        return True
    return False



#repeat until convergence

for numer in range(30):
    print("iteration || ", numer)

    # Initializing number of stars 
    pop_number = 5
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


