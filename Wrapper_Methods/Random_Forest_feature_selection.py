import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def rank(scores,data):
    importances = list(zip(scores, data.columns))
    #print("importances = ", importances)
    sorted_scores = sorted(importances, reverse= True)
    #print("sorted scores = ", sorted_scores)
    return sorted_scores

def get_least(scores):
    features = scores[-1]
    columns  = features[1]
    return columns

#Step 1 : Read the data
if __name__ == "__main__":
    data = pd.read_csv('car_evaluation.csv')
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data.columns = col_names
    Y = data['class']
    X = data.drop(columns= ['class'])
    num_feat = X.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


    encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    print("Xtrain shape = ", X_train.shape)
    feature_to_remove = []
    #Step 2 : Iterative over feature space starting with all features
    while(num_feat >=1):
         print("num_feat ", num_feat)

         #Step 3 :  Train a random forest on a given feature space
         clf = RandomForestClassifier()
         clf.fit(X_train, y_train)
         print("accuracy {} after removing {} ".format(clf.score(X_test,y_test), feature_to_remove) )
         #Step 4 :  get the gini score for each features
         feature_Scores = clf.feature_importances_ 
         #Step 5 : Rank them from highest to lowest 
         ranked_feature_Scores = rank(feature_Scores, X_train)
         #print("ranked  features scores = ", ranked_feature_Scores)
         feature_to_remove = get_least(ranked_feature_Scores)
         #print("features to remove = ", feature_to_remove)
          
         #Step 6 : eliminate last 1 features
         #least_imp_features = ranked_feature_Scores[0:(len(ranked_feature_Scores)-1)]
         #print("least important features = ", least_imp_features)
         #Step 7 : Construct the new dataset with the new features
         X_train = X_train.drop(columns = feature_to_remove)
         X_test = X_test.drop(columns = feature_to_remove)
         
         num_feat = X_train.shape[1]
         #print("num features = ", num_feat)
         

         #Step 8 : repeat
          