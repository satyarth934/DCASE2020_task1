# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 06:46:27 2020

@author: Irtaza
"""

# This function just requires training input_set, testing input_set, training labels, and testing labels
# Trainging input set dimensions (number_of_training_examples, feature size)
# Testing input set dimensions (number_of_testing_examples, feature size)
# Trainging labels dimensions (number_of_training_examples, 1)
# Testing labels dimensions (number_of_testing_examples, 1) 
# Returns the testing accuracy, and confusion matrix 

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

def perform_svm(X_train, X_test, Y_train, Y_test):
    print("Performing SVM...\n")
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    #Predict the response for test dataset
    Y_pred = clf.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    print("Confusion matrix \n", confusion_matrix(Y_test,Y_pred))
    print(classification_report(Y_test,Y_pred))
    return metrics.accuracy_score(Y_test, Y_pred), confusion_matrix(Y_test,Y_pred)
