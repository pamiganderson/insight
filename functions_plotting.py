#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:44:23 2018

@author: pamelaanderson
"""
import matplotlib.pyplot as plt
    
def plot_feature_importance(sorted_series_features, title_str):
    plt.figure()
    sorted_series_features.plot(kind = 'barh', color = 'lightgreen')
    plt.title(title_str)
    plt.show()