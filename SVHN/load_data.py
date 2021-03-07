#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:17:28 2020

@author: guillermogarcia
"""
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters

batch_size = 100

# SVHN dataset. In this case 0 image correspond with 0 label

train_dataset = torchvision.datasets.SVHN(root='../../data/',
                                           split='train', 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.SVHN(root='../../data/',
                                          split = 'test', 
                                          transform=transforms.ToTensor(),
                                         download=True )

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
