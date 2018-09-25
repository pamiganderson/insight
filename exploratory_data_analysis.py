#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:30:57 2018

@author: pamelaanderson
"""


# Scatter plot
import matplotlib.pyplot as plt
df_ad_ev_drug_num.plot(x = 'recall_bool', y = 'serious_count', kind = 'scatter')

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

df = df_ad_ev_drug_num
color_vals = np.array(df['recall_bool'])
color_vals = np.where(color_vals == 1, 'b', 'r')

scatter_matrix(df[['serious_count',
                   'seriousnesscongenitalanomali_count',
                   'seriousnessdeath_count',
                   'seriousnessdisabling_count',
                   'seriousnesshospitalization_count',
                   'seriousnesslifethreatening_count',
                   'seriousnessother_count',
                   'recall_bool']],
                    alpha = 0.8, color = color_vals)


import matplotlib.pyplot as plt
plt.figure()
df_recall_data['classification'][df_recall_data['classification'] == 1].groupby(df_recall_data['report_date'].dt.year).count().plot(kind="bar")
plt.title('Class I Recalls Overtime')
plt.ylabel('Number of Recalls')
plt.xlabel('Year')

plt.figure()
df_recall_data['classification'][df_recall_data['classification'] == 2].groupby(df_recall_data['report_date'].dt.year).count().plot(kind="bar")
plt.title('Class II Recalls Overtime')
plt.ylabel('Number of Recalls')
plt.xlabel('Year')

df_features = df_ad_ev_drug_num[['serious_count', 'seriousnesscongenitalanomali_count',
                                 'seriousnessdeath_count', 'seriousnessdisabling_count',
                                 'seriousnesshospitalization_count', 
                                 'seriousnesslifethreatening_count',
                                 'age_count']]
resp_var = df_ad_ev_drug_num['recall_bool']