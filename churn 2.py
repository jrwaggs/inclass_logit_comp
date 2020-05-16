# -*- coding: utf-8 -*-
"""

@author: jr.waggoner
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering
import sqlalchemy as sa
import configparser
import category_encoders as ce
from sklearn import metrics, linear_model, tree, preprocessing, ensemble, svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, roc_auc_score, precision_score, recall_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta

import warnings
warnings.simplefilter('ignore')   #suppress warning messages 


training = pd.read_csv('Train_Churn.csv', header = 0)

testing = pd.read_csv('Test_Churn.csv', header = 0)


training.head(20)

"""---------------------------------------- initial data cleaning ---------------------------------------------"""
training.dtypes
    
# check for null values
print(training.isnull().sum())


def do_data(df, has_y = True):
    df.columns = map(str.lower, df.columns)
    
    df = df.drop(columns ='totalcharges')
    
    # loop over df and convert all object dtype columns to categorical
    for i in df.columns:
        if df[i].dtype == object:
            df[i] = df[i].astype('category')
             
    df['seniorcitizen'] = df['seniorcitizen'].astype('category')

    # encode outcome and convert to int
  
    if has_y == True: 
        X = df.drop(columns=['churn','customer number'])
        df['churn'] = df['churn'].map({'Yes':1, 'No':0}).astype(int)
        y = df['churn']
        return X, y
    else:
        X = df.drop(columns=['customer number'])
        return X


X, y = do_data(training)

# encode input variabes 

# encoder = ce.CatBoostEncoder()
encoder = ce.OneHotEncoder()     # changed to JS encoder on 1/15
encoder.fit(X,y)
X = encoder.transform(X)

feature_names = list(X.columns)

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30,
                                                    random_state = 5)



"""---------------------------------------- initial modeling ---------------------------------------------"""

#dtc_accuracy_scores

def top_score(score_dict):
    """returns the key/index of the top accuracy score in the scores dictionary used in model testing."""
    
    top_score = 0
    index = 0
    for key,values in score_dict.items():
        if values[-1] > top_score: #last element of values array will always be the score
            top_score = values[-1]
            index = key
    return index


rf_accuracy_scores = {}

iteration = 0
for i in range(2,11):
    for j in range(4,8):
        rf_model = ensemble.RandomForestClassifier(n_estimators = 100,
                                               max_depth = i,
                                               max_features = j)
        rf_model.fit(x_train, y_train)
        score = rf_model.score(x_test, y_test)
        rf_accuracy_scores[iteration] = i,j,score
        iteration += 1

rf_accuracy_scores[top_score(rf_accuracy_scores)]

"""---------------------------------------- maximizing auc---------------------------------------------"""
iteration = 0
for i in range(2,11):
    for j in range(4,8):
        rf_model = ensemble.RandomForestClassifier(n_estimators = 100,
                                               max_depth = i,
                                               max_features = j)
        rf_model.fit(x_train, y_train)
        rf_proba = rf_model.predict_proba(x_test)
        rf_proba = rf_proba[:,1]
        score = roc_auc_score(y_test, rf_proba)
        rf_accuracy_scores[iteration] = i,j,score
        iteration += 1

rf_accuracy_scores[top_score(rf_accuracy_scores)]



rf_model = ensemble.RandomForestClassifier(n_estimators = 1000,
                                            max_depth = 4,
                                            max_features = 6)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_proba = rf_model.predict_proba(x_test)
rf_proba = rf_proba[:,1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_score = rf_model.score(x_test,y_test)
rf_recall = recall_score(y_test, rf_pred)
rf_precision  = precision_score(y_test, rf_pred)
rf_result_mat = confusion_matrix(y_test,rf_pred)

sns.heatmap(rf_result_mat,
            # square = True,
            annot = True,
            fmt = 'd',
            cbar = False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Classifier')
plt.ylim(0,2)
plt.show()

print('Accuracy Score {}'.format(rf_score))
print('AUC {}'.format(rf_auc))
print('Precision score {}'.format(rf_precision))
print('Recall score {}'.format(rf_recall))


Xt = do_data(testing, has_y = False)
Xt = encoder.transform(Xt)

scores = rf_model.predict_proba(Xt)[:,1]

final_scores = pd.DataFrame()
final_scores['Customer Number'] = testing.iloc[:,0]
final_scores['Churn'] = scores

#Exporting .csv 
now = datetime.now().strftime("%m/%d/%Y")

now = datetime.now().strftime('%m/%d/%Y-%H/%M')
filename = ('Waggoner_scores ' + str(now))
filename = filename.replace(" ", "_")
filename = filename.replace("/", "-")

print('exporting file ' + filename + '.csv...' )

final_scores.to_csv(filename + '.csv', header = True)

