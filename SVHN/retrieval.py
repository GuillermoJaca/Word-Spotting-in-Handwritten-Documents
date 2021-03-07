#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:09:31 2020

@author: guillermogarcia
"""
import scipy
from operator import itemgetter

'''
def QbE(image_number,features_test):
    
    distance_hyperparameter= 12
    
    distances=[]
    #First the distance of the query wrt all other images is computed
    for i in range(len(features_test)):
        distances.append(torch.dist(features_test[i],features_test[image_number],2))    
        
    distances= torch.stack(distances)
    
    #After we retrieved the most similar results
    retrieved_search=[]
    for j in range(len(distances)):
        if distances[j]<distance_hyperparameter:
            retrieved_search.append(j)
    return retrieved_search


def QbS(index_search,features_test,labelss):
    
    for i in range(len(features_test)):
        if labelss[i] == index_search:
            image_number= i
            break
        
    return QbE(image_number,features_test)
'''   

def QbE_k_items(imagen_query,k,features_test,labelss):
    
    #Select the imagen query and get a list of [distance to image_queried,label]
    distances_labelled=[]
    for i in range(len(features_test)):
      dist = scipy.spatial.distance.euclidean(features_test[imagen_query], features_test[i])
      distances_labelled.append([dist,labelss[i]])
      
    sorted_distances_labelled= sorted(distances_labelled, key=itemgetter(0))
    list_items= [i[:][1] for i in sorted_distances_labelled ]
    
    return list_items[:k]


def QbS_k_items(index_search,k,features_test,labelss):
    
    for i in range(len(features_test)):
        if labelss[i] == index_search:
            imagen_query= i
            break
        
    return QbE_k_items(imagen_query,k,features_test,labelss)










