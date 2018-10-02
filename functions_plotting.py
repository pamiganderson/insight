#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:44:23 2018

@author: pamelaanderson
"""
import matplotlib.pyplot as plt
import seaborn as sns
    
def plot_feature_importance(sorted_series_features, title_str):
    sns.set()
    plt.figure()
    sorted_series_features.plot(kind = 'barh', color = 'blue')
    plt.title(title_str)
    plt.show()