# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 10:37:48 2021

@author: Matteo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata

def load(filename):
    file = open(filename,'r')
    
    featureList = []
    labelsList = []
    for line in file:
        fields = line.split(',')
        attributes = np.array([float(data) for data in fields[:4]])
        attributes_col = attributes.reshape(attributes.size,1)
        featureList.append(attributes_col)
        labelsList.append(fields[4])
    file.close()
    return np.hstack(featureList), np.array(labelsList, dtype=np.int32)

D_train, L_train = load("Train.txt")
D_test, L_test = load("Test.txt")

features = {
        0: 'Variance',
        1: 'Skewness',
        2: 'Curtosis',
        3: 'Entropy'
        }

D_train_0 = D_train[:,L_train==0]
D_train_1 = D_train[:,L_train==1]
D_test_0 = D_test[:,L_test==0]
D_test_1 = D_test[:,L_test==1]

#for feature in range(4):
 #   plt.figure()
  #  plt.hist(D_train_0[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4)
   # plt.hist(D_train_1[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4)
    #plt.show()
    
"""
GAUSSIANIZATION
"""  
def rank(training_data, dataset):
    ranks = []
    for feature in range(4):
        counts = np.zeros(dataset.shape[1])
        count = 0
        for sample in range(dataset.shape[1]):
            count = np.int64(training_data[feature,:] <= dataset[feature,sample]).sum()
            counts[sample] = (count+1)/(dataset.shape[1]+2)
        ranks.append(counts)
        
    ranks = np.vstack(ranks)
    return ranks

def gaussianize(ranks):
    data_Gaussianized = []
    
    for feature in range(4):
        data_Gaussianized.append(norm.ppf(ranks[feature]))
        
    data_Gaussianized = np.vstack(data_Gaussianized)
    
    return data_Gaussianized

ranks = rank(D_train, D_train)
D_train_Gaussianized = gaussianize(ranks)

D_train_0 = D_train_Gaussianized[:,L_train==0]
D_train_1 = D_train_Gaussianized[:,L_train==1]

ranks = rank(D_train, D_test)
D_test_Gaussianized = gaussianize(ranks)

D_test_0 = D_test_Gaussianized[:,L_test==0]
D_test_1 = D_test_Gaussianized[:,L_test==1]

#for feature in range(4):
 #   plt.figure()
  #  plt.title("Training " + features[feature] + " - Gaussianization")
   # plt.hist(D_train_0[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4) 
    #plt.hist(D_train_1[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4)
    #plt.show()
    #plt.figure()
    #plt.title("Test " + features[feature] + " - Gaussianization")
    #plt.hist(D_test_0[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4)
    #plt.hist(D_test_1[feature,:], bins = 'auto', ec="black", density = True, alpha = 0.4)
    #plt.show()
    
"""
CORRELATION ANALYSIS
"""
correlations = np.corrcoef(D_train_Gaussianized)
print(correlations)
plt.figure()
plt.title("Correlation - dataset")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_0)
print(correlations)
plt.figure()
plt.title("Correlation - class 0")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_1)
print(correlations)
plt.figure()
plt.title("Correlation - class 1")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

"""
PCA
"""
mu = D_train_Gaussianized.mean(1) #mean of columns (dataset mean) #1-D vector
DC = D_train_Gaussianized - mu.reshape((mu.size,1)) #center the data (remove the mean mu from all points)

#covariance matrix 
C = np.dot(DC,DC.T)
C = C / float(DC.shape[1])

m = 3
s, U = np.linalg.eigh(C)
P = U[:, ::-1][:, 0:m]

D_train = np.dot(P.T, D_train)

D_train_0 = D_train[:, L_train == 0]
D_train_1 = D_train[:, L_train == 1]




