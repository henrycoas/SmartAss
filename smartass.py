import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 

def getTrainData():
    # Import
    url = 'trainDatabase.csv'
    dataset = pd.read_csv(url) 

    le = preprocessing.LabelEncoder()
    dataset['topic'] = le.fit_transform(dataset['topic'])
    dataset['subject'] = le.fit_transform(dataset['subject'])
    dataset['format'] = le.fit_transform(dataset['format'])
    dataset['dedication_env'] = le.fit_transform(dataset['dedication_env'])

    return dataset

def getEvalDataset():
    # Import
    url = 'newAssignments.csv'
    dataset = pd.read_csv(url) 

    le = preprocessing.LabelEncoder()
    dataset['topic'] = le.fit_transform(dataset['topic'])
    dataset['subject'] = le.fit_transform(dataset['subject'])
    dataset['format'] = le.fit_transform(dataset['format'])

    return dataset

trainDataset = getTrainData()
evalDataset = getEvalDataset()

# Assign values to the X and y variables:
# id,topic,type,estimated_difficulty,difficulty_after_completion,start_time,finish_time,time_dedicated,grade
# X = trainDataset[['id','topic','type','estimated_difficulty','difficulty_after_completion','start_time','finish_time','time_dedicated','grade']]
X = trainDataset[['id','topic','subject','format','estimated_difficulty']]
y = trainDataset['time_dedicated']
X_eval = evalDataset[['id','topic','subject','format','estimated_difficulty']]

# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_eval = scaler.fit_transform(X_eval)

classifier = MLPClassifier(alpha=1, max_iter=1000)
classifier.fit(X, y) 

y_predict = classifier.predict(X_eval)
print(y_predict)