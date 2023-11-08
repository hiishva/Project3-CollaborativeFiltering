from sklearn import metrics
from sklearn.datasets import fetch_openml
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


#Loads the data from the MNSIT
def LoadData():
    print('Loading data')
    X,y = fetch_openml('mnist_784', version=1,return_X_y=True)
    X = X / 255

    X_train, X_test = X[:60000], X[60000:]
    label_train, label_test = y[:60000], y[60000:]
    
    print('Data loaded')
    return X_test,X_train,label_test,label_train

def KNearestNeighbors(X_train ,X_test,label_test,label_train):
    print('BEGIN K NEAREST NEIGHBORS CLASSIFIER')
    #Parameters that are being changed are N_neighbors, weight, and the p parameter
    #the values that are chosen are N_Neighbors: [3,5,7] weights: [distance, uniform], and p: [1,2]
    KNNClassifier = KNeighborsClassifier(n_neighbors=7,weights='distance',p=2)
    KNNClassifier.fit(X_train,label_train)

    pred = KNNClassifier.predict(X_test)

    accuracy = metrics.accuracy_score(label_test,pred)
    print('parameters are: n_neighbor = 7, weight = distance, p = 2')
    print('KNN Accuracy Score: {}'.format(accuracy))
    print('KNN Error Rate: {}'.format(1-accuracy))
    print('END K NEAREST NEIGHBORS CLASSIFIER')
    print('------------------------------------------')
    return 
def SVM(X_train, X_test, label_test, label_train):
    print('BEGIN SVM CLASSIFIER')
    
    # HYPERPARAMETERS
    kernel = ['linear', 'rbf', 'poly']
    CVals = [0.1,10]
    gammaVals = ['scale','auto']

    for k in kernel:
        for c in CVals:
            for g in gammaVals:
                svm = SVC(kernel=k, C=c, gamma=g)
                svm.fit(X_train, label_train)
                
                pred = svm.predict(X_test)
                accuracy = metrics.accuracy_score(label_test,pred)
                print('parameters are: kernel = {}, C value = {}, gamma = {}'.format(k,c,g))
                print('SVM Accuracy Score: {}'.format(accuracy))
                print('SVM Error Rate: {}'.format(1-accuracy))
                print('-----------')

    # k = 'rbf'
    # c = 0.1
    # g = 'auto'
    # svm = SVC(kernel=k, C=c, gamma=g)
    # svm.fit(X_train,label_train)

    # pred = svm.predict(X_test)
    # accuracy = metrics.accuracy_score(label_test,pred)

    # print('parameters are: kernel = {}, C value = {}, gamma = {}'.format(k,c,g))
    # print('SVM Accuracy Score: {}'.format(accuracy))
    # print('SVM Error Rate: {}'.format(1-accuracy))

    return
warnings.filterwarnings("ignore", category=FutureWarning)

X_test, X_train, label_test, label_train = LoadData()
#KNearestNeighbors(X_train ,X_test,label_test,label_train)
SVM(X_train ,X_test,label_test,label_train)
