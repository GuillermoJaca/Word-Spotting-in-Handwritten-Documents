#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:36:59 2020

@author: guillermogarcia
"""
import torch 
import torch.nn as nn
import torchvision

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Train the model
def train_model(num_epochs,model,cross_entropy_,optimizer,train_loader):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = cross_entropy_(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))