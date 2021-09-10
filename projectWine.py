# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:58:24 2021

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
        attributes = np.array([float(data) for data in fields[:11]])
        attributes_col = attributes.reshape(attributes.size,1)
        featureList.append(attributes_col)
        labelsList.append(fields[11])
    file.close()
    return np.hstack(featureList), np.array(labelsList, dtype=np.int32)

def vcol(inputV):
    return inputV.reshape((inputV.size,1))

def plot_features(dataset_0, dataset_1, features, dataset_type):
    for feature in range(dataset_0.shape[0]):
        plt.figure()
        plt.title(dataset_type + " " + features[feature])
        plt.hist(dataset_0[feature,:], bins = 30, ec="black", density = True, alpha = 0.4)
        plt.hist(dataset_1[feature,:], bins = 30, ec="black", density = True, alpha = 0.4)
        plt.show()

D_train, L_train = load("Train.txt")
D_test, L_test = load("Test.txt")

features = {
        0: 'Fixed Acidity',
        1: 'Volatile Acidity',
        2: 'Citric Acid',
        3: 'Residual Sugar',
        4: 'Chlorides',
        5: 'Free Sulfur Dioxide',
        6: 'Total Sulfur Dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
        }

D_train_0 = D_train[:,L_train==0]
D_train_1 = D_train[:,L_train==1]
D_test_0 = D_test[:,L_test==0]
D_test_1 = D_test[:,L_test==1]

#plot_features(D_train_0, D_train_1, features, 'Training')
#plot_features(D_test_0, D_test_1, features, 'Test')

"""
GAUSSIANIZATION
"""  
def rank(training_data, dataset):
    ranks = []
    for feature in range(dataset.shape[0]):
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
    
    for feature in range(ranks.shape[0]):
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

#plot_features(D_train_0, D_train_1, features, 'Gaussianized Training')
#plot_features(D_test_0, D_test_1, features, 'Gaussianized Test')

"""
CORRELATION ANALYSIS
"""
def compute_correlation(dataset, label):
    correlations = np.corrcoef(dataset)
    #print(correlations)
    plt.figure()
    plt.title("Correlation - " + label)
    plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
    plt.show()
    
    
correlations = np.corrcoef(D_train_Gaussianized)
#print(correlations)
plt.figure()
plt.title("Correlation - dataset")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_0)
#print(correlations)
plt.figure()
plt.title("Correlation - class 0")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

correlations = np.corrcoef(D_train_1)
#print(correlations)
plt.figure()
plt.title("Correlation - class 1")
plt.imshow(correlations, cmap='gist_yarg', vmin=-1, vmax=1)
plt.show()

"""
PCA with m=10
"""
def PCA(dataset, m):
    mu = dataset.mean(1) #mean of columns (dataset mean) #1-D vector
    DC = dataset - mu.reshape((mu.size,1)) #center the data (remove the mean mu from all points)
    
    #covariance matrix 
    C = np.dot(DC,DC.T)
    C = C / float(DC.shape[1])
    
    m = 10
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    
    return np.dot(P.T, dataset)
    
D_train_G_PCA = PCA(D_train_Gaussianized, 10)
D_train_PCA = PCA(D_train, 10)

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
    nTrain = int(D.shape[1]*8.0/10.0) #take 80% of the original dataset as training data
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) #random order for data indexes
    idxTrain = idx[0:nTrain] #nTrain indexes are for Training
    idxTest = idx[nTrain:] #others for Testing
    DTR = D[:, idxTrain] #training data
    DTE = D[:, idxTest] #evaluation data
    LTR = L[idxTrain] #training labels
    LTE = L[idxTest] #evaluation labels
    return (DTR, LTR), (DTE, LTE)
    
def MVG_minDCF(DTR, LTR, DTE, LTE):
    DTR0 = DTR[:, LTR==0] #training samples of class 0
    DTR1 = DTR[:, LTR==1] #training samples of class 1
    
    #ML means
    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)
    
    
    #ML covariances
    C0 = np.dot((DTR0-vcol(mu0)),(DTR0-vcol(mu0)).T) / DTR0.shape[1]
    C1 = np.dot((DTR1-vcol(mu1)),(DTR1-vcol(mu1)).T) / DTR1.shape[1]
    
    C0 = np.cov(DTR0)
    C1 = np.cov(DTR1)
    
    like0 = logpdf_GAU_ND(DTE, mu0, C0)
    like1 = logpdf_GAU_ND(DTE, mu1, C1)
    
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    
    compute_minDCF(llr,LTE,0.5,1,1)
    compute_minDCF(llr,LTE,0.1,1,1)
    compute_minDCF(llr,LTE,0.9,1,1)
    
def naiveBayes_minDCF(DTR, LTR, DTE, LTE):
    DTR0 = DTR[:, LTR==0] #training samples of class 0
    DTR1 = DTR[:, LTR==1] #training samples of class 1
    
    #ML means
    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)
    
    #ML covariances
    C0 = (np.dot((DTR0-vcol(mu0)),(DTR0-vcol(mu0)).T) / DTR0.shape[1]) * np.eye(DTR0.shape[0])
    C1 = (np.dot((DTR1-vcol(mu1)),(DTR1-vcol(mu1)).T) / DTR1.shape[1]) * np.eye(DTR1.shape[0])
    
    like0 = logpdf_GAU_ND(DTE, mu0, C0)
    like1 = logpdf_GAU_ND(DTE, mu1, C1)
    
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    
    compute_minDCF(llr,LTE,0.5,1,1)
    compute_minDCF(llr,LTE,0.1,1,1)
    compute_minDCF(llr,LTE,0.9,1,1)

def tiedCovariance_minDCF(DTR, LTR, DTE, LTE):
    DTR0 = DTR[:, LTR==0] #training samples of class 0
    DTR1 = DTR[:, LTR==1] #training samples of class 1
    
    #ML means
    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)
    
    #ML covariances
    C0 = np.dot((DTR0 - vcol(mu0)), (DTR0 - vcol(mu0)).T)
    C1 = np.dot((DTR1 - vcol(mu1)), (DTR1 - vcol(mu1)).T)
    C = (C0+C1) / DTR.shape[1]
    
    like0 = logpdf_GAU_ND(DTE, mu0, C)
    like1 = logpdf_GAU_ND(DTE, mu1, C)
    
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    
    compute_minDCF(llr,LTE,0.5,1,1)
    compute_minDCF(llr,LTE,0.1,1,1)
    compute_minDCF(llr,LTE,0.9,1,1)
    
def K_Fold_minDCF(D, L, K):
    n = int(D.shape[1] / K) #samples per fold
    
    best_minDCF_05 = 1.0   #best achieved minDCF for pi=0.5
    mu_0_star_05 = 0         #parameter mu0 of the best model for pi=0.5
    mu_1_star_05 = 0         #parameter mu1 of the best model for pi=0.5
    C_star_05 = np.zeros((D.shape[0], D.shape[0]))  #parameter C of the best tied covariance model for pi=0.5
    C_0_star_05 = np.zeros((D.shape[0], D.shape[0])) #parameter C0 of the best model for pi=0.5
    C_1_star_05 = np.zeros((D.shape[0], D.shape[0])) #parameter C1 of the best model for pi=0.5
    
    best_minDCF_01 = 1.0   #best achieved minDCF for pi=0.1
    mu_0_star_01 = 0         #parameter mu0 of the best model for pi=0.1
    mu_1_star_01 = 0         #parameter mu1 of the best model for pi=0.1
    C_star_01 = np.zeros((D.shape[0], D.shape[0]))  #parameter C of the best tied covariance model for pi=0.1
    C_0_star_01 = np.zeros((D.shape[0], D.shape[0])) #parameter C0 of the best model for pi=0.1
    C_1_star_01 = np.zeros((D.shape[0], D.shape[0])) #parameter C1 of the best model for pi=0.1
    
    best_minDCF_09 = 1.0   #best achieved minDCF for pi=0.9
    mu_0_star_09 = 0         #parameter mu0 of the best model for pi=0.9
    mu_1_star_09 = 0         #parameter mu1 of the best model for pi=0.9
    C_star_09 = np.zeros((D.shape[0], D.shape[0]))  #parameter C of the best tied covariance model for pi=0.9
    C_0_star_09 = np.zeros((D.shape[0], D.shape[0])) #parameter C0 of the best model for pi=0.9
    C_1_star_09 = np.zeros((D.shape[0], D.shape[0])) #parameter C1 of the best model for pi=0.9
 
    """
    MVG
    """    
    for i in range(K):
        DTR = []
        LTR = []
        DTE = []
        LTE = []
        
        for j in range(K):
            if(i == j):
                DTE.append(D[:,j*n:(j+1)*n])
                LTE.append(L[j*n:(j+1)*n])
            else:
                DTR.append(D[:,j*n:(j+1)*n])
                LTR.append(L[j*n:(j+1)*n])
                
        DTE = np.hstack(DTE).ravel().reshape(D.shape[0],n) #evaluation fold
        LTE = np.hstack(LTE).ravel()                        #evaluation labels
        DTR = np.hstack(DTR).ravel().reshape(D.shape[0],(K-1)*n) #K-1 training folds
        LTR = np.hstack(LTR).ravel()                            #training labels
    
        DTR0 = DTR[:, LTR==0] #training samples of class 0
        DTR1 = DTR[:, LTR==1] #training samples of class 1
        

        #ML means
        mu0 = DTR0.mean(1)
        mu1 = DTR1.mean(1)
        
        
        #ML covariances
        C0 = np.dot((DTR0-vcol(mu0)),(DTR0-vcol(mu0)).T) / DTR0.shape[1]
        C1 = np.dot((DTR1-vcol(mu1)),(DTR1-vcol(mu1)).T) / DTR1.shape[1]
        
        C0 = np.cov(DTR0)
        C1 = np.cov(DTR1)
        
        like0 = logpdf_GAU_ND(DTE, mu0, C0)
        like1 = logpdf_GAU_ND(DTE, mu1, C1)
        
        S = np.vstack([like0,like1])
        llr = S[1] - S[0]
        
        current_minDCF = compute_minDCF(llr,LTE,0.5,1,1)
        if(current_minDCF < best_minDCF_05):
            best_minDCF_05 = current_minDCF
            mu_0_star_05 = mu0
            mu_1_star_05 = mu1
            C_0_star_05 = C0
            C_1_star_05 = C1
            
        current_minDCF = compute_minDCF(llr,LTE,0.1,1,1)
        if(current_minDCF < best_minDCF_01):
            best_minDCF_01 = current_minDCF
            mu_0_star_01 = mu0
            mu_1_star_01 = mu1
            C_0_star_01 = C0
            C_1_star_01 = C1
        
        current_minDCF = compute_minDCF(llr,LTE,0.9,1,1)
        if(current_minDCF < best_minDCF_09):
            best_minDCF_09 = current_minDCF
            mu_0_star_09 = mu0
            mu_1_star_09 = mu1
            C_0_star_09 = C0
            C_1_star_09 = C1
    
    like0 = logpdf_GAU_ND(DTE, mu_0_star_05, C_0_star_05)
    like1 = logpdf_GAU_ND(DTE, mu_1_star_05, C_1_star_05)
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    print("final minDCF:")
    compute_minDCF(llr,LTE,0.5,1,1)
    
    like0 = logpdf_GAU_ND(DTE, mu_0_star_01, C_0_star_01)
    like1 = logpdf_GAU_ND(DTE, mu_1_star_01, C_1_star_01)
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    print("final minDCF:")
    compute_minDCF(llr,LTE,0.1,1,1)
    
    like0 = logpdf_GAU_ND(DTE, mu_0_star_09, C_0_star_09)
    like1 = logpdf_GAU_ND(DTE, mu_1_star_09, C_1_star_09)
    S = np.vstack([like0,like1])
    llr = S[1] - S[0]
    print("final minDCF:")
    compute_minDCF(llr,LTE,0.9,1,1)

    
    
"""
Multivariate Gaussian Classifier
"""

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_Gaussianized, L_train)
print("minDCF for MVG model (Gaussianized - no PCA):")
MVG_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_G_PCA, L_train)
print("minDCF for MVG model (Gaussianized - 10-PCA:")
MVG_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train, L_train)
print("minDCF for MVG model (non Gaussianized - no PCA):")
MVG_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_PCA, L_train)
print("minDCF for MVG model (non Gaussianized - 10-PCA):")
MVG_minDCF(DTR,LTR,DTE,LTE)

"""
Naive Bayes Classifier
"""

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_Gaussianized, L_train)
print("minDCF for Naive Bayes model (Gaussianized - no PCA):")
naiveBayes_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_G_PCA, L_train)
print("minDCF for Naive Bayes model (Gaussianized - 10-PCA:")
naiveBayes_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train, L_train)
print("minDCF for Naive Bayes model (non Gaussianized - no PCA):")
naiveBayes_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_PCA, L_train)
print("minDCF for Naive Bayes model (non Gaussianized - 10-PCA):")
naiveBayes_minDCF(DTR,LTR,DTE,LTE)

"""
Tied Covariances Classifier
"""

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_Gaussianized, L_train)
print("minDCF for Tied Covariance model (Gaussianized - no PCA):")
tiedCovariance_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_G_PCA, L_train)
print("minDCF for Tied Covariance model (Gaussianized - 10-PCA:")
tiedCovariance_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train, L_train)
print("minDCF for Tied Covariance model (non Gaussianized - no PCA):")
tiedCovariance_minDCF(DTR,LTR,DTE,LTE)

(DTR, LTR), (DTE, LTE) = split_db_singleFold(D_train_PCA, L_train)
print("minDCF for Tied Covariance model (non Gaussianized - 10-PCA):")
tiedCovariance_minDCF(DTR,LTR,DTE,LTE)

print("minDCF for K-fold cross-validation, Gaussianized - no PCA")
K_Fold_minDCF(D_train_Gaussianized, L_train, 5)
print("minDCF for K-fold cross-validation, non Gaussianized - no PCA")
K_Fold_minDCF(D_train, L_train, 5)
print("minDCF for K-fold cross-validation, Gaussianized - 10-PCA")
K_Fold_minDCF(D_train_G_PCA, L_train, 5)
print("minDCF for K-fold cross-validation, non Gaussianized - 10-PCA")
K_Fold_minDCF(D_train_PCA, L_train, 5)

