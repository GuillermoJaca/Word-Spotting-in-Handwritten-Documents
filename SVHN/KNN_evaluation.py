#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:43:11 2020

@author: guillermogarcia
"""

from sklearn.neighbors import KNeighborsClassifier

def KNN_eval(features_train,features_test,labels_train_total,labelss):
    #Make classification with all the images of the test set and check whether
    #it is accurate enough
    
    classifier = KNeighborsClassifier(n_neighbors=1)  
    classifier.fit(features_train, labels_train_total)
    
    y_pred = classifier.predict(features_test) 
    
    correct_cont= 0
    NumImages= 10000
    for i in range(NumImages):
      if ((y_pred[i] == labelss[i]) ): correct_cont += 1
    total_cont = correct_cont/NumImages
    
    return total_cont