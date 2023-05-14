import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

""" 
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
] """

def getTrainData():
    # Import
    url = 'trainDatabase.csv'
    dataset = pd.read_csv(url) 

    # Label treatment
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

    # Label treatment
    le = preprocessing.LabelEncoder()
    dataset['topic'] = le.fit_transform(dataset['topic'])
    dataset['subject'] = le.fit_transform(dataset['subject'])
    dataset['format'] = le.fit_transform(dataset['format'])

    return dataset

trainDataset = getTrainData()
evalDataset = getEvalDataset()

# Assign values to the X and y variables:
X = trainDataset[['id','topic','subject','format','estimated_difficulty']]
y = trainDataset['time_dedicated']
X_eval = evalDataset[['id','topic','subject','format','estimated_difficulty']]

# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_eval = scaler.fit_transform(X_eval)

#para = {'alpha':[1,2,3,5,20]} 
#CV = GridSearchCV(classifier, para, cv=5, n_jobs=-1)
classifier = MLPClassifier(alpha=1, max_iter=1000)
classifier.fit(X, y) 

y_predict = classifier.predict(X_eval)
print(y_predict)