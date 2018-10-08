#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:44:23 2018

@author: pamelaanderson
"""
import matplotlib.pyplot as plt
import seaborn as sns
    
def plot_feature_importance(sorted_series_features, title_str):
    """ Plot feature importance from tree models """
    sns.set()
    plt.figure()
    sorted_series_features.plot(kind = 'barh', color = 'blue')
    plt.title(title_str)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout(pad=2.0, w_pad=5.0, h_pad=1.0)
    plt.show()
    
    
def plot_serious_events(df_merge_class):
    """ plot serious events over 2013 and 2014 """
    colors = {1: '#00ACFF', 0: '#FF001F'}
    zone_name = dict({1: 'Lower Risk', 0: 'Higher Risk'})
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    grouped = df_merge_class.groupby('classify_risk')
    for key, group in  grouped:
        group.plot(ax=ax1, kind='scatter', x='serious_count_pre', y='serious_count',
                   label=zone_name[key], color=colors[key], s=5, alpha=0.5)
    ax1.set_ylabel('# Serious Events 2014')
    ax1.set_xlabel('# Serious Events 2013')
    ax1.set_title('Serious Events Plotted By Risk Class')
    #ax1.legend_.remove()
    plt.legend(frameon=False, loc='upper left', ncol=1, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(pad=2.0, w_pad=5.0, h_pad=1.0)
    
def exploratory_plot(df):
    from pandas.plotting import scatter_matrix

    color_vals = np.array(df['risk_class'])
    color_vals = np.where(color_vals == 1, 'b', 'r')
    
    scatter_matrix(df[['sum_tot_bene',
                       'sum_tot_claim',
                       'sum_tot_dosage',
                       'sum_tot_spend',
                       'risk_class']],
                        alpha = 0.8, color = color_vals)

def plot_manuf_vs_generic(df_spending_2014):
    df_piv_2014 = pd.pivot_table(df_spending_2014, index='generic_name', 
                                 values='manufacturer',
                                 aggfunc = 'count')
    df_piv_2014 = df_piv_2014.reset_index()
    
    plt.figure()
    df_merge_2014.plot(kind='scatter', x = 'manufacturer', y = 'serious_count')
    plt.xlabel('Number of Manufacturers')
    plt.ylabel('Number of Adverse Events')
    plt.title('2014 Q1 Adverse Events vs. Manufacturer #')