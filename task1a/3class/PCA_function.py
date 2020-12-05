# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 06:55:29 2020

@author: Irtaza
"""

# This inputs of this function are Dataset in a from (number of examples, number of features), their labels,
# percentage information that controls how many eigen vectors use for transformation, and test set percentage 
# means what percentage of total data you want to include in the testing set.
# Perfroms the PCA and returns the tranfromed version training set and testing set.
# More on the information_amount input, it ranges from 0 to 1. 0 means include no eigen vector, and 1 means
# include all eigen vectors. Eigen values depicts the information present in the corresponding eigen vector. 
# This information_amount parameter tells PCA to include eigen vectors upto the point so that the 
# information retained after tranformation/ total information becomes equal to information_amount.
# Simply, we can say information_amount = information retained after tranformation/ total information

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_PCA(Dataset, label, information_amount, test_set_percentage):
    print("Train test splitting...\n")
    X_train, X_test, Y_train, Y_test = train_test_split( Dataset,label, test_size=test_set_percentage, random_state=0)        
    
    print("standardizing dataset...\n")
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Performing PCA...\n")
    # Make an instance of the Model
    pca = PCA(information_amount)
    # fit PCA on training data
    pca.fit(X_train)
    
    print("Number of features now: ", pca.n_components_)
    print("\nProjecting on the eigen vectors...\n")
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, Y_train, Y_test