#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:28:13 2020

@author: guillermogarcia
"""
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def features_test_extraction(test_loader,model):
    model.eval()
    pred = []
    with torch.no_grad():
        for images, labels in test_loader:
                images = images.to(device)
                outputs = model.features(images)
                pred.append(outputs)
    
    pred_= torch.stack(pred[:260])
    pred__=torch.reshape(pred_,(26000,128))
    pred___= torch.cat((pred__,pred[260]),0)
    
    features_test=pred___
    return features_test


def features_train_extraction(train_loader,model):
    model.eval()
    pred_train = []
    labels_train_total=[]
    with torch.no_grad():
        for images, labels in train_loader:
                images = images.to(device)
                labels_train = labels.to(device)
                outputs = model.features(images)
                pred_train.append(outputs)
                labels_train_total += labels_train.tolist()
    
    pred_train_= torch.stack(pred_train[:732])
    pred_train__=torch.reshape(pred_train_,(73200,128))
    pred_train___= torch.cat((pred_train__,pred_train[732]),0)
    
    features_train=pred_train___
    
    return features_train, labels_train_total
