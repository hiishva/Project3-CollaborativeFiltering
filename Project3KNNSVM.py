from sklearn import metrics
from sklearn.datasets import fetch_openml
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#Load data
def LoadData():
    print('Loading data')
    X,y = fetch_openml('mnist_784', version=1,return_X_y=True)
    X = X / 255

    X_train, X_test = X[:60000], X[60000:]
    label_train, label_test = y[:60000], y[60000:]
    
    print('Data loaded')
    return X_test,X_train,label_test,label_train

def KNearstNeighbors(X_train ,X_test,label_test,label_train):
    print('beginning K Nearest Neighbors')
    KNNClassifier = KNeighborsClassifier()
    KNNClassifier.fit(X_train,label_train)
    predict = KNNClassifier.predict(X_test)
    accuracy = metrics.accuracy_score(label_test,predict)
    print('KNN Accuracy Score: {}'.format(accuracy))

    return 

warnings.filterwarnings("ignore", category=FutureWarning)

X_test, X_train, label_test, label_train = LoadData()
KNearstNeighbors(X_train ,X_test,label_test,label_train)
