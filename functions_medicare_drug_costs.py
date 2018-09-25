#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:22:59 2018

@author: pamelaanderson
"""
import numpy as np
from numpy import inf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

########## Set up spending dataframe ##########
def read_spending_csv(path, file_name, year):
    df_spending = pd.read_csv(path + file_name)
    df_spending_noheader = df_spending.drop(0,axis=0)
    
    year_column_dict = {'2012' : [3, 10],
                        '2013' : [10, 17],
                        '2014' : [17, 24]}
        
    df_spending_year = pd.concat([df_spending_noheader.iloc[:,0:3],
                                  df_spending_noheader.iloc[:,year_column_dict[year][0]:year_column_dict[year][1]]], axis=1)
    df_spending_year.columns = df_spending.iloc[0][0:10]
    return df_spending_year
    
def format_str_and_numerics(df_spending_2014):
    val_list =[]
    for i, val in enumerate(df_spending_2014['generic_name']):
        if val[-1] == " ":
            val_list.append(df_spending_2014.iloc[i]['generic_name'][:-1])
        else:
            val_list.append(df_spending_2014.iloc[i]['generic_name'][:])
    df_spending_2014['generic_name'] = pd.Series(val_list)
    
    val_list =[]
    for i, val in enumerate(df_spending_2014['brand_name']):
        if val[-1] == " ":
            val_list.append(df_spending_2014.iloc[i]['brand_name'][:-1])
        else:
            val_list.append(df_spending_2014.iloc[i]['brand_name'][:])
    df_spending_2014['brand_name'] = pd.Series(val_list)
            
    df_spending_2014['brand_name'] = df_spending_2014['brand_name'].str.lower()
    df_spending_2014['generic_name'] = df_spending_2014['generic_name'].str.lower()

    # Make the raised amount total usd column into numeric values
    list_col_names_to_convert_num = ['total_spending', 'total_dosage_units', 
                                     'total_claims', 'total_beneficiaries',
                                     'average_spending_per_dosage_unit']
    for k in list_col_names_to_convert_num:
        series_tmp = df_spending_2014[k]
        series_tmp = series_tmp.str.replace(",","")
        series_tmp = series_tmp.str.replace("$","")
        series_tmp = series_tmp.str.replace(" ","")
        series_tmp = series_tmp.apply(pd.to_numeric)
        df_spending_2014[k] = series_tmp

    return df_spending_2014
    
def create_ad_df_by_brand(df_ad_data_from_sql_q1):
    # Pivot table for adverse events by generic_name
    df_ad_data_from_sql_q1['drug_generic_name'] = df_ad_data_from_sql_q1['drug_generic_name'].str.lower()
    df_ad_data_from_sql_q1['drug_brand_name'] = df_ad_data_from_sql_q1['drug_brand_name'].str.lower()
    
    df_piv_adv_2014_brand = pd.pivot_table(df_ad_data_from_sql_q1,
                                           index=['drug_brand_name','drug_generic_name'],
                                           values='serious_count',
                                           aggfunc=np.sum)
    df_piv_adv_2014_brand = df_piv_adv_2014_brand.reset_index()
    return df_piv_adv_2014_brand

def merge_spending_df_and_ad_df(df_spending_2014, df_piv_adv_2014_brand):
    df_spending_brand = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['total_spending', 
                                                 'total_dosage_units', 
                                                 'total_claims', 
                                                 'total_beneficiaries',
                                                 'average_spending_per_dosage_unit',
                                                 'manufacturer'],
                                       aggfunc = np.sum)
    df_spending_brand = df_spending_brand.reset_index()
        
    # Merge with adverse events numbers
    df_merge_2014 = df_spending_brand.merge(df_piv_adv_2014_brand, left_on='brand_name',
                                      right_on='drug_brand_name',
                                      how = 'left')
    df_merge_2014['serious_per_bene'] = 100*(df_merge_2014['serious_count']/df_merge_2014['total_beneficiaries'])
    return df_merge_2014

def feature_and_generic_label(df):
    # If no adverse events reported, assume 0
    df = df.fillna(0)
    df = df.drop(['drug_brand_name', 'drug_generic_name'], axis=1)
    
    df_na = df #df.dropna().sort_values(by='generic_name')
    risk_class_list = []
    diff_cost_per_dose_list = []
    df_na = df_na.reset_index()
    # Find contingency table for each generic
    # format [[brand_ad_ev, brand_bene], [generic_ad_ev, generic_bene]]
    for i, val in enumerate(df_na['generic_name']):
        if df_na.iloc[i]['brand_name'] == val:
            # GENERIC NEG = -1
            risk_class_list.append(-1)
            diff_cost_per_dose_list.append(df_na.iloc[i]['average_spending_per_dosage_unit'])
        else:
            # BRAND POS = 1
            risk_class_list.append(1)
            diff_cost_per_dose_list.append(df_na.iloc[i]['average_spending_per_dosage_unit'])
    
    risk_series = pd.Series(risk_class_list).replace(np.inf, np.nan)
    risk_series = risk_series.replace(-np.inf, np.nan)
    df_na['risk_class'] = risk_series
    df_na['diff_cost_per_dose'] = diff_cost_per_dose_list
    
    df_piv_merge_generic_risk = pd.pivot_table(df_na, index = ['generic_name',
                                                               'risk_class'],
        values = ['serious_count', 'total_beneficiaries', 'total_claims',
                  'total_dosage_units', 'total_spending', 'manufacturer',
                  'diff_cost_per_dose'], aggfunc = np.sum).reset_index()
    df_piv_merge_generic_risk_max = pd.pivot_table(df_na, index = ['generic_name',
                                                               'risk_class'],
        values = ['diff_cost_per_dose'], aggfunc = max).reset_index()    
    df_piv_merge_generic_risk_min = pd.pivot_table(df_na, index = ['generic_name',
                                                           'risk_class'],
        values = ['diff_cost_per_dose'], aggfunc = min).reset_index()
    df_piv_merge_generic_risk['min_cost_per_dose'] = df_piv_merge_generic_risk_min['diff_cost_per_dose']
    df_piv_merge_generic_risk['max_cost_per_dose'] = df_piv_merge_generic_risk_max['diff_cost_per_dose']
    return df_piv_merge_generic_risk

########## CLASSIFICATION ##########
def classify_generic_risk(df_piv_merge_generic_risk):    
    p_val_chi_sq = []
    index_list = []
    for i, val in enumerate(df_piv_merge_generic_risk['generic_name'][:-1]):
        if val == df_piv_merge_generic_risk['generic_name'][i+1]:
            # create contingency table
            # first entry will be for brand, second will be for generic
            #if df_piv_merge_generic_risk.iloc[i]['risk_class'] == 1
            obs = [[df_piv_merge_generic_risk.iloc[i]['serious_count'], 
                   df_piv_merge_generic_risk.iloc[i]['total_beneficiaries']],
                    [df_piv_merge_generic_risk.iloc[i+1]['serious_count'],
                    df_piv_merge_generic_risk.iloc[i+1]['total_beneficiaries']]]
            if (((obs[0][0] + obs[1][0]) == 0.0) | (obs[0][1] == 0) | (obs[1][1] == 0)):
                p_val_chi_sq.append(0)
                index_list.append(i)
            else:
                chi2, p, dof, expected = chi2_contingency(obs)
                # Check if first entry is a generic (-1) and then if the risk the ratio is higher, generic is high risk
                if (df_piv_merge_generic_risk.iloc[i]['risk_class'] == -1):
                    if (obs[0][0]/obs[1][0] > obs[0][1]/obs[1][1]):
                        p_val_chi_sq.append(-1*p)
                        index_list.append(i)
                    else:
                        p_val_chi_sq.append(p)
                        index_list.append(i)
                # Check if first entry is a brand (1) and then if the risk the ratio is higher, generic is high risk
                elif (df_piv_merge_generic_risk.iloc[i]['risk_class'] == 1):
                    # the   
                    if (obs[0][0]/obs[1][0] > obs[0][1]/obs[1][1]):
                        p_val_chi_sq.append(p)
                        index_list.append(i)
                    else:
                        p_val_chi_sq.append(-1*p)
                        index_list.append(i)
        else:
            p_val_chi_sq.append(0)
            index_list.append(i)
    
    p_val_chi_sq.append(0)

    df_piv_merge_generic_risk['p_val_chisq'] = pd.Series(p_val_chi_sq)
    #series_max_min = df_piv_merge_generic_risk['risk_class'].multiply(df_piv_merge_generic_risk['min_cost_per_dose'])
    risk_boolean = df_piv_merge_generic_risk['risk_class']
    risk_boolean[risk_boolean == 1] = True
    risk_boolean[risk_boolean == -1] = False
    series_max_min = df_piv_merge_generic_risk['min_cost_per_dose']
    series_max_min[risk_boolean] = df_piv_merge_generic_risk['max_cost_per_dose']
    df_piv_merge_generic_risk['dose_price_range'] = series_max_min
    df_class_generic = pd.pivot_table(df_piv_merge_generic_risk, index = ['generic_name'],
                                      values = ['risk_class', 'total_beneficiaries',
                                                'total_claims', 'total_dosage_units',
                                                'total_spending', 'p_val_chisq', 'dose_price_range'],
                                      aggfunc = np.sum)
    classify_risk = df_class_generic['p_val_chisq']
    classify_risk[classify_risk>=0]=1
    classify_risk[classify_risk<0]=0

    df_class_generic['classify_risk'] = classify_risk

#    df_class_generic['risk_class'][df_class_generic['risk_class'] > 0] = 1
#    df_class_generic['risk_class'][df_class_generic['risk_class'] < 0] = 0
    return df_class_generic

def find_sim_bene_num(df_class_generic):
    num_thres = 5000
    df_class_generic_ratio = df_class_generic[abs(df_class_generic['sum_tot_bene']) < num_thres]
    return df_class_generic_ratio

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

    




