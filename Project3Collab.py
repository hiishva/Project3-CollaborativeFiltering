import numpy as np
import glob
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def ReadFromFolder(directoryPath):
    print('in readfromfolder')
    testDataset = pd.DataFrame()
    trainDataset = pd.DataFrame()
    for fileName in sorted(glob.glob(directoryPath+'/*.txt')):
        if 'Test' in fileName:
            testDataset = pd.read_csv(fileName,low_memory=False,header=None)
            print('added to testdata')
        elif 'Train' in fileName:
            trainDataset = pd.read_csv(fileName,low_memory=False,header=None)
            print('added to train data')
    return testDataset, trainDataset

def CollabFilter(train, test):
    print('starting collab filtering')
    #Similarity Matrix
    #Training 
    ds2 = train.pivot(index = 'custId', columns = 'movieId', values = 'rating')
    usrMean = ds2.mean(axis = 1)
    corMtrx = ds2.T.corr().fillna(0)
    print('cormatrix created')
    
    #prediction time!!
    rateDiff = ds2.sub(usrMean, axis=0) #standardization
    mom = usrMean.mean() #pop mean
    usrCount = ds2.index 
    movCount = ds2.columns
    print('about to preds')
    preds = []

    for i in range(len(test)):
        print('first for loop!')
        movk,useri = [int(test.iat[i,0]), int(test.iat[i,1])]
        
        #check if user i is in user count
        if useri in usrCount:
            print('check if i rated smthng')
            useriMean = usrMean.loc(useri) #use the users mean
        else:
            useriMean = mom #use the pop mean
        if movk in movCount:
            print('checking the movies')
            rateDiffj = rateDiff.loc[:,movk]
            usrjnotk =rateDiffj[rateDiffj.isnull()].index

            weight = corMtrx.loc[useri,:].copy()
            weight.loc[usrjnotk] = None
            sumWeight = weight.abs().sum()

            #collaborative ratings
            if sumWeight != 0:
                print('check the weight')
                collRate = (weight * rateDiffj).sum() / sumWeight
            else:
                collRate = 0
        else:
            collRate = 0 
        rateik = useriMean + collRate
        preds.append(round(max(min(rateik, 5), 1), 1))
    return preds

directorypath = os.getcwd()
print(directorypath)
testData,trainData = ReadFromFolder(directorypath)
testData.columns = ['movieId', 'custId', 'rating']
trainData.columns = ['movieId','custId','rating']
preds = CollabFilter(trainData,testData)
print(preds[0])