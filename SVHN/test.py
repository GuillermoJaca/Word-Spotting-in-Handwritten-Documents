#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:13:30 2020

@author: guillermogarcia
"""
import torch

def test_model(model,test_loader,device):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    pred = []
    labelss= []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            labelss += labels.tolist()
            pred += predicted.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print('Test Accuracy of the model on the 26032 test images: {} %'.format(100 * correct / total))
    
    return pred,labelss