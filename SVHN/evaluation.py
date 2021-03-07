#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:25:18 2020

@author: guillermogarcia
"""
import numpy as np
import scipy
from operator import itemgetter

def rs_one_image(imagen_query,features_test,y_test):
    '''
    Return a list with 1 in case of hit in the retrieved search and 0 otherwise
    '''
    
    #Select the imagen query and get a list of [distance to image_queried,label]
    distances_labelled=[]
    for i in range(len(features_test)):
      dist = scipy.spatial.distance.euclidean(features_test[imagen_query], features_test[i])
      distances_labelled.append([dist,y_test[i]])
      
    #Order list, so the most similar images are at the beginning
    sorted_distances_labelled= sorted(distances_labelled, key=itemgetter(0))
    
    #Get a list with 1 in case of hit and 0 otherwise
    rs=[]
    for i in range(len(features_test)):
      if sorted_distances_labelled[i][1] == y_test[imagen_query]: rs.append(1)
      else: rs.append(0)
    
    return rs


def rs_x_images(features_test,y_test,x):
    '''
    Same as rs_one_image but a list of lists of x images
    '''
    rs_total = []
    for i in range(x):
      rs_total.append(rs_one_image(imagen_query=i,features_test=features_test,y_test=y_test))
    
    return rs_total



def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def AP(imagen_query,features_test,y_test):
    
    return average_precision(rs_one_image(imagen_query,features_test,y_test))
    
def mAP(features_test,y_test,x):
    
    return mean_average_precision(rs_x_images(features_test, y_test,x))

