#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:22:13 2020

@author: guillermogarcia
"""

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(labelss,pred):
    array = confusion_matrix(labelss,pred)
    df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],
                      columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    
    """We can observe clearly the numbers that are misclasified more often (looking at the out-diagonal terms):
    3 and 7, 0 and 6, 5 and 6, 6 and 8, 7 and 1, 2 and 7. They are numbers similar in its shape so they can be easily misclasified.
    """
