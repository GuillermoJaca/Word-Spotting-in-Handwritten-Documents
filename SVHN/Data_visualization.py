#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:26:49 2020

@author: guillermogarcia
"""
import matplotlib.pyplot as plt

def data_visualize(train_loader):
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape
    
    plt.imshow(example_data[1].view(32,32,3))
    print(example_targets[1])