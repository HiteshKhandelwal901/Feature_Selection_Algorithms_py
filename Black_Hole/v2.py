import os
import random
import re
import time
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import List
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, label_ranking_loss, precision_score, label_ranking_average_precision_score
import matplotlib.pyplot as plt
from filters import univariate_feature_elimination
from skmultilearn.adapt import MLkNN
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

NUM_PROCESSES = 8 # Number of experiments you want to process paralley. ==> eventually ends up dependent on the max cpu and tasks as well for getting upper bound
DATA_PATH = './Data/scene.csv'
REPORT_PATH = './reports'
NUM_LABELS = 6 # no of labels 
NUM_STARS = 20 # How many stars per iteration?
NUM_EXPERIMENTS = 20 # How many experiments? ==> Useful for stability testing and remove randomness bias
NUM_ITERATIONS = 50 # How many iterations per experiment
STAR_DIM = 294 # Max feature number
LAMBDA = 0.0002 # Regularization parameter.
MOVEMENT_PROBABILITY = 0.5 # probability of making a move

class Star:
    """
    In the context of feature selection, a star is the list of feature indices which will be 
    used for the predictive model. 
    Feature set representation:
    Unlike the more readble/visualizable format of having a binary representation for stars,
    we use the compressed version of the feature set where we just keep the list of the feastures which are 
    considered to be selected for a given star. 
    This helps in saving memory and time(theorotically!)
    Note: This represenation has forced us to reimagine/formulate some of the calculations which are more obvious in binary represenation
    However no generality is lost in doing so and resuls should be the same without any compromise to achieve the above said efficiency(if any!!)
    """

    def __init__(self, active_features: List):
        """
        activate features is a list of features which are active for this star
        Here we use a KNN classifer however in theory we can change it to any other classifier by modifying the model class instanstiaion and 
        methods: classifier_fit and classifier_predict(If needed)
        """

        self.active_features = active_features
        self.fitness = 0
        self.loss = 0
        #self.classifier = KNeighborsClassifier(n_neighbors=10)
        self.classifier = MLkNN(k=10)

    def classifier_fit(self, X: np.ndarray, Y: np.ndarray):
        self.classifier.fit(X, Y)

    def classifier_predict(self, X_test):
        """
        Predicts the op class/classes for given ip examples
        """
        return self.classifier.predict(X_test)

    def calculate_fitness(self, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray):
        """
        Fitness is calculated by Hammings Loss penalized by number of attribues selected
        """
        # Assign very low fitness if no features are selected as it is not a viable solution
        if len(self.active_features) == 0:
            return -float("inf")
        self.classifier_fit(X_train[:, self.active_features], Y_train)
        pred = self.classifier_predict(X_train[:, self.active_features]).toarray()
        self.loss = hamming_loss(Y_train, pred)
        self.ranking_loss = label_ranking_loss(Y_train, pred)
        self.average_precision = label_ranking_average_precision_score(Y_train, pred)
        self.fitness = (1-self.loss)/(1 + LAMBDA * len(self.active_features))

    def move(self, blackhole_features: List):
        """
        Probabilitsically add or remove features which are not included/included in the current Star comapred with the BlackHole
        Since we are using compressed feature representation, we will make a probabilitic move only if there is a mistach between selection
        status of a feature in blackhole and the star under consideration. This means we add a feature if its is in blackhole but not in star 
        and delete a feature if it is in star and not in blackhole
        """

        for feature in blackhole_features:
            if feature not in self.active_features and random.random() < MOVEMENT_PROBABILITY:
                self.active_features.append(feature)
        for feature in self.active_features:
            if feature not in blackhole_features and random.random() < MOVEMENT_PROBABILITY:
                self.active_features.remove(feature)

    def star_distance(self, star_1, star_2):
        """
        In essence, this performs euclidean distance between 2 binary vectors.
        Since we are using compressed representation of the features selected, 
        XOR of the 2 feature sets will result in the mismatch features and hence euclidean distance
        ends up as sqrt(no of mismatched features). This is because element wise difference between 2 feature set 
        can be at max 1(and minimum can be 0). 
        """

        return math.sqrt(len(list(set(star_1) ^ set(star_2))))


    def is_proximity_star(self, blackhole_features: List, radius):
        """
        A star is said to in proximty if it lies within a radius of "radius" units from a blackhole
        """
        return self.star_distance(self.active_features, blackhole_features) < radius

    def reinitialize(self):
        """
        Random selection of features
        """
        self.active_features = [feature_idx for feature_idx in range(len(self.active_features)) if random.random() < 0.5]

class BlackHole:
    """
    class to run Black Hole algorith for multilabel classification

    """
    def __init__(self, num_stars: int, num_iterations: int, star_dim: int):
        self.num_stars = num_stars
        self.num_iterations = num_iterations
        self.star_dim = star_dim
        self.global_best_fitness = -float("inf")
        self.global_best_star = None
        self.feature_histogram = {idx:0 for idx in range(self.star_dim)}

    def generate_random_stars(self):
        """
        randomly activates a feature.
        Note: Refer to the class Star docstring to understand our represenattion of a feature set.
        """
        self.stars = [Star([var_idx for var_idx in range(self.star_dim) if random.random() > 0.5]) for _ in range(self.num_stars)]
            
    def evaluate_fitness(self, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray):
        """
        Move non blackhole star towards star
        If star is too close to blackhole, randomly reinitialize it(Stephen Hawkings might have argued!)
        """

        blackhole = None
        blackhole_fitness = -float("inf")
        total_fitness = 0
        # Evaluate the fitness for each  star
        for star in self.stars:
            star.calculate_fitness(X_train, X_test, Y_train, Y_test)
            # Keep track of local best solution
            if star.fitness > blackhole_fitness:
                blackhole = star
                blackhole_fitness = star.fitness
            if star.fitness > self.global_best_fitness:
                self.global_best_fitness = star.fitness
                self.global_best_star = star
            total_fitness += star.fitness
        rejection_radius = blackhole.fitness / total_fitness
        # Move all non-blackhole stars
        for star in self.stars:
            if star == blackhole:
                pass
            else:
                star.move(blackhole.active_features)
        # reinitialize star if its too close to balckhole
        """
        How close to too close?
        ==> Build a sphere around the black hole based on the blackhole fitness relatibe to toat fitness of all stars
        ==> if the star enters this sphere, reinitialise the star
        """
        for star in self.stars:
            if star == blackhole:
                pass
            else:
                if star.is_proximity_star(blackhole.active_features, rejection_radius):
                    star.reinitialize()
        # Update Statistics related to features included across iterations
        # self.update_histogram()
        # self.unused_features()

    def update_histogram(self):
        """
        Updates the frequency count of a fearture across the iteraions
        """
        for star in self.stars:
            for feature in star.active_features:
                self.feature_histogram[feature] += 1
    
    def unused_features(self):
        """
        Returns a list of features which never got selected ever in any iteration
        """
        self.unused= [feature for feature in self.feature_histogram.keys()  if self.feature_histogram[feature] == 0]
    

    def single_experiment(self, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray):
        for iter in range(self.num_iterations):
            if iter % 10 == 0:
                print("*"*100)
                print("Iteration:", iter)
            self.evaluate_fitness(X_train, X_test, Y_train, Y_test)



def read_data(data_loc: str, seed: int):
    # TODO Handle path better. Currently just a string representation
    # TODO Decouple logic of feature eleimination to avoid unnecessary duplication
    data = pd.read_csv(data_loc)
    X, Y = data.iloc[:, :-NUM_LABELS],  data.iloc[:, -NUM_LABELS:]
    # TODO Does order of the following 2 steps matter? 
    X = univariate_feature_elimination(X,Y,15)
    X = MinMaxScaler().fit_transform(X.values)
    Y = Y.values
    #print("seed = ", seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=seed)
    """
    pd.DataFrame(X_train).to_csv('./data/train_data.csv', index=False)
    pd.DataFrame(Y_train).to_csv('./data/train_label.csv', index=False)
    pd.DataFrame(X_test).to_csv('./data/test_data.csv', index=False)
    pd.DataFrame(Y_test).to_csv('./data/test_label.csv', index=False)
    """
    return X_train, X_test, Y_train, Y_test

def create_report(all_metrics: List[dict]):
    """
    Create report of training and test metrics and other statistics
    """
    # TODO Can we loopify the plot?
    # TODO Spacing and beautification of graphs not done. Labels are not readable and overlapping
    # TODO Add summary statistics to the excel
    
    report_df = pd.DataFrame(all_metrics)
    print("*"*100)
    print("Report:", report_df)
    print("*"*100)
    # Create the report directory if it does not exist
    if not os.path.exists(REPORT_PATH):
        print("Creating Report directory", REPORT_PATH)
        os.mkdir(REPORT_PATH)
    report_df.to_excel(os.path.join(REPORT_PATH, 'report.xlsx'))
    fig, ax = plt.subplots( nrows=2, ncols=4, sharex='col', sharey='row')
    ax[0, 0].hist(report_df['hamming_loss'])
    ax[0, 0].set_title("Training Hamming Loss")
    ax[0, 1].hist(report_df['ranking_loss'])
    ax[0, 1].set_title("Training Ranking Loss")
    ax[0, 2].hist(report_df['average_precision'])
    ax[0, 2].set_title("Training Average Precision")
    ax[0, 3].hist(report_df['num_features'])
    ax[0, 3].set_title("# Features")
    ax[1, 0].hist(report_df['test_hamming_loss'])
    ax[1, 0].set_title("Test Hamming Loss")
    ax[1, 1].hist(report_df['test_ranking_loss'])
    ax[1, 1].set_title("Test Ranking Loss")
    ax[1, 2].hist(report_df['test_average_precision'])
    ax[1, 2].set_title("Test Average Precision")
    ax[1, 3].hist(report_df['num_features'])
    ax[1, 3].set_title("# Features")
    fig.savefig(os.path.join(REPORT_PATH, 'profile.png'))
    plt.close(fig)  

def single_process(experiment_id: int):
        """
        Id associated with experiment, just for book keeping/seed generation
        """
        # TODO: Handle seed such that experiments are repeatable with same seed. Use same seed across different libraries to ensure this 
        random.seed(experiment_id)
        seed = random.randint(1, 1000)
        print(100 * "*")
        print("Running experiment number: ", experiment_id)
        print(100 * "*")
        # TODO: 1 Time feature selection need not happen everytime. Decouple the logic so that 1st stage feature sel;ection happens just
        X_train, X_test, Y_train, Y_test = read_data(DATA_PATH, seed)
        bh = BlackHole(NUM_STARS, NUM_ITERATIONS, X_train.shape[1])
        bh.generate_random_stars()
        bh.single_experiment(X_train, X_test, Y_train, Y_test)
        print("Experiment:", experiment_id)
        print("Best Black Hole Hamming Loss:", bh.global_best_star.loss)
        print("*"*100)
        pred = bh.global_best_star.classifier.predict(X_test[:, bh.global_best_star.active_features]).toarray()
        metrics = dict()
        metrics['hamming_loss'] = bh.global_best_star.loss
        metrics['ranking_loss'] = bh.global_best_star.ranking_loss
        metrics['average_precision'] = bh.global_best_star.average_precision
        metrics['fitness'] = bh.global_best_star.fitness
        metrics['best_feature_set'] = bh.global_best_star.active_features
        metrics['num_features'] = len(bh.global_best_star.active_features)
        metrics['test_hamming_loss'] = hamming_loss(Y_test, pred)
        metrics['test_ranking_loss'] = label_ranking_loss(Y_test, pred)
        metrics['test_average_precision'] = label_ranking_average_precision_score(Y_test, pred)
        return metrics 

def run_experiments(num_experiments: int):
    """
    Perform Black Hole Algorithms multiple item with different random seed each time
    Processess the runs parallely depending upon the num of processes/experiment
    """
    # TODO:   Add checks to ensure processes are auto killed after processed to avoid stale processes
    experiment_list = list(range(num_experiments))
    with Pool(processes=min(num_experiments, NUM_PROCESSES, cpu_count())) as pool:
        res = pool.map(single_process, experiment_list)
        #print("test loss = ", res['test_hamming_loss'])    
        print(res)
    create_report(res)


def main():
    run_experiments(NUM_EXPERIMENTS)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Time Taken:", time.time()-start_time)