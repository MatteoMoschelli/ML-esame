# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 10:37:48 2021

@author: Matteo
"""

import numpy as np
import matplotlib.pyplot as plt
import math
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

def vcol(inputV):
    return inputV.reshape((inputV.size,1))

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
plt.figure()
plt.title("Correlation - dataset")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_0)
plt.figure()
plt.title("Correlation - class 0")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_1)
plt.figure()
plt.title("Correlation - class 1")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

"""
PCA with m=3 and m=2
"""
mu = D_train_Gaussianized.mean(1) #mean of columns (dataset mean) #1-D vector
DC = D_train_Gaussianized - mu.reshape((mu.size,1)) #center the data (remove the mean mu from all points)

#covariance matrix 
C = np.dot(DC,DC.T)
C = C / float(DC.shape[1])

m = 3
s, U = np.linalg.eigh(C)
P = U[:, ::-1][:, 0:m]

D_train_PCA_3 = np.dot(P.T, D_train_Gaussianized)

D_train_PCA_3_0 = D_train_PCA_3[:, L_train == 0]
D_train_PCA_3_1 = D_train_PCA_3[:, L_train == 1]

m = 2
s, U = np.linalg.eigh(C)
P = U[:, ::-1][:, 0:m]

D_train_PCA_2 = np.dot(P.T, D_train_Gaussianized)

D_train_PCA_3_0 = D_train_PCA_2[:, L_train == 0]
D_train_PCA_3_1 = D_train_PCA_2[:, L_train == 1]

"""
minDCF
"""
def logpdf_GAU_ND(X, mu, C):
    mu = vcol(mu)
    Y = []
    for col in X.T:
        x = np.array(col).reshape(col.shape[0],1)
        M = x.shape[0]
        sign, log_det = np.linalg.slogdet(C)
        y = -(M/2)*np.log(2*np.pi) -0.5 * log_det -0.5 * np.dot(np.dot((x-mu).T, np.linalg.inv(C)), (x-mu))
    
        Y.append(y)
    return np.hstack(Y)

def compute_OBD(llr, labels, treshold):
  optimal_bayes_decisions = np.zeros([2,2])
  for i in range(len(llr)):
      if llr[i] > treshold:
        optimal_bayes_decisions[1, labels[i]]+=1
      else:
        optimal_bayes_decisions[0, labels[i]]+=1
  return optimal_bayes_decisions

def compute_FNR_FPR(optimal_bayes_decision):
  FNR = (optimal_bayes_decision[0,1])/(optimal_bayes_decision[0,1] + optimal_bayes_decision[1,1])
  FPR = (optimal_bayes_decision[1,0])/(optimal_bayes_decision[0,0] + optimal_bayes_decision[1,0])
  return (FNR, FPR)

def compute_DCF(*args):
  llr = args[0]
  labels = args[1]
  prior_class_probabilities = args[2]
  C_fn = args[3]
  C_fp = args[4]
  if (len(args) == 5):
    treshold = -np.log((args[2]*args[3])/((1-args[2])*args[4]))
  else:
    treshold = args[5]
  optimal_bayes_decision = compute_OBD(llr, labels, treshold)
  (FNR, FPR) = compute_FNR_FPR(optimal_bayes_decision)
  return prior_class_probabilities*C_fn*FNR + (1-prior_class_probabilities)*C_fp*FPR

def compute_normalizeDCF(*args):
    return compute_DCF(*args) / (min(args[2]*args[3], (1-args[2])*args[4]))

def compute_minDCF(llr, labels, prior_class_probabilities, C_fn, C_fp):
  nDCF = math.inf
  tresholds = np.hstack(([-math.inf], llr.ravel(), [math.inf]))
  tresholds.sort()
  for treshold in tresholds:
    current_nDCF = compute_normalizeDCF(llr, labels, prior_class_probabilities, C_fn, C_fp, treshold)
    if (current_nDCF < nDCF): 
      nDCF = current_nDCF
  print(nDCF)
  return nDCF

def split_db_singleFold(D, L, seed=0):
    nTrain = int(D.shape[1]*4.0/5.0) #take 80% of the original dataset as training data
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) #random order for data indexes
    idxTrain = idx[0:nTrain] #nTrain indexes are for Training
    idxTest = idx[nTrain:] #others for Testing
    DTR = D[:, idxTrain] #training data
    DTE = D[:, idxTest] #evaluation data
    LTR = L[idxTrain] #training labels
    LTE = L[idxTest] #evaluation labels
    return (DTR, LTR), (DTE, LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_Gaussianized, L_train)


#dataset mean and class means
mu = DTR.mean(1)
DTR0 = DTR[:, LTR==0] #training samples of class 0
DTR1 = DTR[:, LTR==1] #training samples of class 1

#ML means
mu0 = DTR0.mean(1)
mu1 = DTR1.mean(1)

#ML covariances
C0 = np.dot((DTR0-vcol(mu0)),(DTR0-vcol(mu0)).T) / DTR0.shape[1]
C1 = np.dot((DTR1-vcol(mu1)),(DTR1-vcol(mu1)).T) / DTR1.shape[1]

like0 = logpdf_GAU_ND(DTE, mu0, C0)
like1 = logpdf_GAU_ND(DTE, mu1, C1)


llr = np.log(like1 / like0).ravel()

minDCF_MVG_full = compute_minDCF(llr,LTE,0.5,1,1)

"""
Naive Bayes Classifier
"""
#ML means
mu0 = DTR0.mean(1)
mu1 = DTR1.mean(1)

#ML covariances
C0 = (np.dot((DTR0-vcol(mu0)),(DTR0-vcol(mu0)).T) / DTR0.shape[1]) * np.eye(DTR0.shape[0])
C1 = (np.dot((DTR1-vcol(mu1)),(DTR1-vcol(mu1)).T) / DTR1.shape[1]) * np.eye(DTR1.shape[0])

like0 = logpdf_GAU_ND(DTE, mu0, C0)
like1 = logpdf_GAU_ND(DTE, mu1, C1)

llr = np.log(like1 / like0).ravel()

minDCF_MVG_diag = compute_minDCF(llr,LTE,0.5,1,1)

"""
tied covariances
"""
mu0 = DTR0.mean(1)
mu1 = DTR1.mean(1)

C0 = np.dot((DTR0 - vcol(mu0)), (DTR0 - vcol(mu0)).T)
C1 = np.dot((DTR1 - vcol(mu1)), (DTR1 - vcol(mu1)).T)

C = (C0+C1) / DTR.shape[1]

like0 = logpdf_GAU_ND(DTE, mu0, C)
like1 = logpdf_GAU_ND(DTE, mu1, C)

llr = np.log(like1 / like0).ravel()

minDCF_MVG_tied = compute_minDCF(llr,LTE,0.5,1,1)












