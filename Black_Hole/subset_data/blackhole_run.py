import pandas as pd 
from utility import hamming_scoreCV


data = pd.read_csv('subset_data/Standalone_scene_88.csv')

Y = data[['Beach','Sunset','FallFoliage','Field','Mountain','Urban']]
X = data.drop(columns= Y)

score, clf, correct, incorrect = hamming_scoreCV(X, Y)