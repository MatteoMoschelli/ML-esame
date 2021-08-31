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

for feature in range(4):
    plt.figure()
    plt.hist(D_train_0[feature,:], bins = 'auto', density = True, alpha = 0.4)
    plt.hist(D_train_1[feature,:], bins = 'auto', density = True, alpha = 0.4)
    plt.show()
    
"""
GAUSSIANIZATION
"""

ranks = []
ranks2 = []

for feature in range(4):
    ranks.append((rankdata(D_train[feature,:], 'ordinal') + 1) / (D_train.shape[1] + 2))

    counts = np.zeros(D_train.shape[1])
    count = 0
    for sample in range(D_train.shape[1]):
        count = np.int64(D_train[feature,:] <= D_train[feature,sample]).sum()
        counts[sample] = (count+1)/(D_train.shape[1]+2)
    ranks2.append(counts)
    print(np.abs(ranks2[feature]-ranks[feature]).mean())
    
    
ranks = np.vstack(ranks)
ranks2 = np.vstack(ranks2)
D_train_Gaussianized = []

for feature in range(4):
    D_train_Gaussianized.append(norm.ppf(ranks2[feature]))
    
D_train_Gaussianized = np.vstack(D_train_Gaussianized)

D_train_0 = D_train_Gaussianized[:,L_train==0]
D_train_1 = D_train_Gaussianized[:,L_train==1]

for feature in range(4):
    plt.figure()
    plt.title(features[feature])
    plt.hist(D_train_0[feature,:], bins = 'auto', density = True, alpha = 0.4) 
    plt.hist(D_train_1[feature,:], bins = 'auto', density = True, alpha = 0.4)
    plt.show()
    
    plt.figure()
    plt.title(features[feature])
    plt.hist(D_train_Gaussianized[feature,:], bins = 'auto', density=True, alpha=0.4)
    plt.show()
    
"""
CORRELATION ANALYSIS
"""
correlations = np.corrcoef(D_train_0)
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

"""
PCA
"""
mu = D_train.mean(1) #mean of columns (dataset mean) #1-D vector
DC = D_train - mu.reshape((mu.size,1)) #center the data (remove the mean mu from all points)

#covariance matrix 
C = np.dot(DC,DC.T)
C = C / float(DC.shape[1])

m = 3
s, U = np.linalg.eigh(C)
P = U[:, ::-1][:, 0:m]

D_train = np.dot(P.T, D_train)

D_train_0 = D_train[:, L_train == 0]
D_train_1 = D_train[:, L_train == 1]




