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
                                                 'total_beneficiaries'],
                                       aggfunc = np.sum)
    df_spending_brand_min = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['average_spending_per_dosage_unit'],
                                       aggfunc = min).reset_index()
    df_spending_brand_max = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['average_spending_per_dosage_unit',
                                                 'manufacturer'],
                                       aggfunc = max).reset_index()
    df_spending_brand = df_spending_brand.reset_index()
    df_spending_brand['min_price_per_dose'] = df_spending_brand_min['average_spending_per_dosage_unit']
    df_spending_brand['max_price_per_dose'] = df_spending_brand_max['average_spending_per_dosage_unit']
    df_spending_brand['manufacturer'] = df_spending_brand_max['manufacturer']
    # Merge with adverse events numbers
    df_merge_2014 = df_spending_brand.merge(df_piv_adv_2014_brand, left_on='brand_name',
                                      right_on='drug_brand_name',
                                      how = 'left')
    df_merge_2014['serious_per_bene'] = 100*(df_merge_2014['serious_count']/df_merge_2014['total_beneficiaries'])
    return df_merge_2014

def merge_spending_df_tot_and_ad_df(df_spending_2014, df_piv_adv_2014_brand):
    df_spending_brand = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['total_spending', 
                                                 'total_dosage_units', 
                                                 'total_claims', 
                                                 'total_beneficiaries'],
                                       aggfunc = np.sum)
    df_spending_brand_mean = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['percent_change_spending', 
                                                 'percent_change_dosage', 
                                                 'percent_change_claims', 
                                                 'percent_change_bene',
                                                 'percent_change_avg_spending_per_dose'],
                                       aggfunc = np.nanmean)
    df_spending_brand = df_spending_brand.merge(df_spending_brand_mean,
                                                left_index=True,
                                                right_index=True,
                                                how='inner')
    df_spending_brand_min = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['current_avg_spending_per_dose'],
                                       aggfunc = min).reset_index()
    df_spending_brand_max = pd.pivot_table(df_spending_2014, index = ['brand_name', 
                                                                  'generic_name'],
                                       values = ['current_avg_spending_per_dose'],
                                       aggfunc = max).reset_index()
    df_spending_brand = df_spending_brand.reset_index()
    df_spending_brand['min_price_per_dose'] = df_spending_brand_min['current_avg_spending_per_dose']
    df_spending_brand['max_price_per_dose'] = df_spending_brand_max['current_avg_spending_per_dose']

    # Merge with adverse events numbers
    df_merge_2014 = df_spending_brand.merge(df_piv_adv_2014_brand, left_on='brand_name',
                                      right_on='drug_brand_name',
                                      how = 'left')
    df_merge_2014['serious_per_bene'] = 100*(df_merge_2014['serious_count']/df_merge_2014['total_beneficiaries'])
    return df_merge_2014

def label_brand_generic(df):
    # If no adverse events reported, assume 
    df = df.reset_index(drop=True)
    df = df.drop(['drug_brand_name', 'drug_generic_name'], axis=1)
    df['generic_compare'] = df['generic_name'].str.replace('-', ' ')
    df['generic_compare'] = df['generic_compare'].str.replace('with ', '')
    df['generic_compare'] = df['generic_compare'].str.replace('/', ' ')
    
    df['brand_compare'] = df['brand_name'].str.replace('-', ' ')
    df['brand_compare'] = df['brand_compare'].str.replace('with ', '')
    df['brand_compare'] = df['brand_compare'].str.replace('/', ' ')
    
    df_na = df.fillna(0) #df.dropna().sort_values(by='generic_name')
    risk_class_list = []
    # Find contingency table for each generic
    # format [[brand_ad_ev, brand_bene], [generic_ad_ev, generic_bene]]
    for i, val in enumerate(df_na['generic_compare']):
        if ((df_na.iloc[i]['brand_compare'] == val) | (df_na.iloc[i]['brand_compare'] in val) |
                (val in df_na.iloc[i]['brand_compare'])):
            # GENERIC NEG = -1
            risk_class_list.append(-1)
            
        else:
            # BRAND POS = 1
            risk_class_list.append(1)
    
    risk_series = pd.Series(risk_class_list).replace(np.inf, np.nan)
    risk_series = risk_series.replace(-np.inf, np.nan)
    df_na['risk_class'] = risk_series
    df['risk_class'] = risk_series
    
    # Drop columns that are redunant from name matching
    df_na = df_na.drop(['generic_compare', 'brand_compare'], axis = 1)
    df = df.drop(['generic_compare', 'brand_compare'], axis = 1)
    
    df_class_generic_count = pd.pivot_table(df, index = ['generic_name'],
                                            values = ['risk_class'], aggfunc = 'count')
    df_class_generic_count.rename(columns={'risk_class' : 'risk_count'}, inplace=True)
    df = df.merge(df_class_generic_count, right_index=True, left_on = 'generic_name', how='inner')
    return df

def feature_and_generic_label(df):  
    
    df_piv_merge_generic_risk = pd.pivot_table(df, index = ['generic_name',
                                                               'risk_class'],
        values = ['serious_count', 'total_beneficiaries', 'total_claims',
                  'total_dosage_units', 'total_spending', 'serious_count_pre',
                  'serious_count_pre_pre', 'serious_range'],
                  aggfunc = np.sum)
    df_piv_merge_generic_risk_mean = pd.pivot_table(df, index = ['generic_name',
                                                               'risk_class'],
        values = ['percent_change_spending', 'percent_change_dosage', 'percent_change_claims', 
                  'percent_change_bene', 'percent_change_avg_spending_per_dose', 'risk_count'],
                  aggfunc = np.nanmean)
    df_piv_merge_generic_risk = df_piv_merge_generic_risk.merge(df_piv_merge_generic_risk_mean,
                                                                left_index=True,
                                                                right_index=True,
                                                                how='inner').reset_index()
    df_piv_merge_generic_risk_max = pd.pivot_table(df, index = ['generic_name',
                                                               'risk_class'],
        values = ['max_price_per_dose'], aggfunc = np.nanmax).reset_index()    
    df_piv_merge_generic_risk_min = pd.pivot_table(df, index = ['generic_name',
                                                           'risk_class'],
        values = ['min_price_per_dose'], aggfunc = np.nanmin).reset_index()
    df_piv_merge_generic_risk['min_cost_per_dose'] = df_piv_merge_generic_risk_min['min_price_per_dose']
    df_piv_merge_generic_risk['max_cost_per_dose'] = df_piv_merge_generic_risk_max['max_price_per_dose']
    df_piv_merge_generic_risk = df_piv_merge_generic_risk.fillna(0)

    return df_piv_merge_generic_risk

def find_num_manuf_and_change(df_spending_pre, df_spending_post):
    df_spending_2manuf_pre = pd.pivot_table(df_spending_pre, index = 'generic_name',
                                         values = 'total_spending',
                                         aggfunc = 'count')
    df_spending_2manuf_post = pd.pivot_table(df_spending_post, index = 'generic_name',
                                         values = 'total_spending',
                                         aggfunc = 'count')
    df_manufacturer = df_spending_2manuf_post-df_spending_2manuf_pre
    df_manufacturer.columns = ['increase_manuf']
    df_manufacturer['total_manuf'] = df_spending_2manuf_post['total_spending']
    return df_manufacturer

########## CLASSIFICATION ##########
def classify_generic_risk(df_piv_merge_generic_risk):   
    df_piv_merge_generic_risk = df_piv_merge_generic_risk.sort_values(by='generic_name')
    p_val_chi_sq = []
    chi_sq_val = []
    index_list = []
    for i, val in enumerate(df_piv_merge_generic_risk['generic_name'][:-1]):
        if val == df_piv_merge_generic_risk['generic_name'][i+1]:
            # create contingency table
            # first entry will be for brand, second will be for generic
            ad_ev_1 = df_piv_merge_generic_risk.iloc[i]['serious_count_pre_pre']
            tot_1 = df_piv_merge_generic_risk.iloc[i]['total_beneficiaries']
            ad_ev_2 = df_piv_merge_generic_risk.iloc[i+1]['serious_count_pre_pre']
            tot_2 = df_piv_merge_generic_risk.iloc[i+1]['total_beneficiaries']
            
            if df_piv_merge_generic_risk.iloc[i]['risk_class'] == -1:
                # obs matriox [[generic_adr, generic_non_adr], [brand_adr brand_non_adr] ]
                obs = [[ad_ev_1, (tot_1-ad_ev_1)],
                        [ad_ev_2, (tot_2-ad_ev_2)]]
                
                row_1 = (ad_ev_1+(tot_1-ad_ev_1))
                row_2 = (ad_ev_2+(tot_2-ad_ev_2))
                col_1 = ad_ev_1 + ad_ev_2
                col_2 = (tot_1-ad_ev_1) + (tot_1-ad_ev_1)
                tot = ad_ev_1 + ad_ev_2 + (tot_1-ad_ev_2) + (tot_2-ad_ev_1)
                # Calculate expected matrix
                exp_obs = [[(row_1*col_1)/tot, (row_2*col_1)/tot], [(row_1*col_2)/tot, (row_2*col_2)/tot]]

            else:
                # obs matriox [[[generic_adr, generic_non_adr], brand_adr brand_non_adr]]
                obs = [[ad_ev_2, (tot_2-ad_ev_2)],
                        [ad_ev_1, (tot_1-ad_ev_1)]]
                row_1 = (ad_ev_2+(tot_2-ad_ev_2))
                row_2 = (ad_ev_1+(tot_1-ad_ev_1))
                col_1 = ad_ev_2 + ad_ev_1
                col_2 = (tot_2-ad_ev_2) + (tot_1-ad_ev_1)
                tot = ad_ev_1 + ad_ev_2 + (tot_1-ad_ev_2) + (tot_2-ad_ev_1)
                # Calculate expected matrix
                exp_obs = [[(row_1*col_1)/tot, (row_2*col_1)/tot], [(row_1*col_2)/tot, (row_2*col_2)/tot]]
                
            if (((obs[0][0] + obs[1][0]) == 0.0) | (obs[0][1] <= 0) | (obs[1][1] <= 0)):
                p_val_chi_sq.append(1)
                chi_sq_val.append(1)
                index_list.append(i)
            else:
                chi2, p, dof, expected = chi2_contingency(obs)
                if obs[0][0] > exp_obs[0][0]:
                    p_val_chi_sq.append(p)
                else:
                    p_val_chi_sq.append(p)
                chi_sq_val.append(chi2)
                index_list.append(i)
        else:
            p_val_chi_sq.append(0)
            chi_sq_val.append(0)
            index_list.append(i)
    
    p_val_chi_sq.append(0)
    chi_sq_val.append(0)

    df_piv_merge_generic_risk['p_val_chisq'] = pd.Series(p_val_chi_sq)
    df_piv_merge_generic_risk['chi_sq_val'] = pd.Series(chi_sq_val)
    #series_max_min = df_piv_merge_generic_risk['risk_class'].multiply(df_piv_merge_generic_risk['min_cost_per_dose'])
    risk_boolean = df_piv_merge_generic_risk[['generic_name','risk_class']]
    df_both_contain = pd.pivot_table(df_piv_merge_generic_risk, index='generic_name',
                                     values = 'risk_class', aggfunc=np.sum)
    both_list = df_both_contain[df_both_contain['risk_class']==0]
    both_list = list(both_list.index.values)
    
    list_max_min = []
    list_range_total_bene = []
    list_range_total_claim =[]
    list_range_total_dosage_units = []
    list_range_total_spending = []
    for i, val in enumerate(risk_boolean['generic_name']):
        if val in both_list:
            if risk_boolean.iloc[i]['risk_class'] == 1:
                # For brand name drugs
                list_max_min.append(df_piv_merge_generic_risk.iloc[i]['max_cost_per_dose'])
                list_range_total_bene.append(df_piv_merge_generic_risk.iloc[i]['total_beneficiaries'])
                list_range_total_claim.append(df_piv_merge_generic_risk.iloc[i]['total_claims'])
                list_range_total_dosage_units.append(df_piv_merge_generic_risk.iloc[i]['total_dosage_units'])
                list_range_total_spending.append(df_piv_merge_generic_risk.iloc[i]['total_spending'])
            elif risk_boolean.iloc[i]['risk_class'] == -1:
               # For generic drugs
                list_max_min.append(-1*df_piv_merge_generic_risk.iloc[i]['min_cost_per_dose'])
                list_range_total_bene.append(-1*df_piv_merge_generic_risk.iloc[i]['total_beneficiaries'])
                list_range_total_claim.append(-1*df_piv_merge_generic_risk.iloc[i]['total_claims'])
                list_range_total_dosage_units.append(-1*df_piv_merge_generic_risk.iloc[i]['total_dosage_units'])
                list_range_total_spending.append(-1*df_piv_merge_generic_risk.iloc[i]['total_spending'])

        else:
            list_max_min.append(0)
            list_range_total_bene.append(0)
            list_range_total_claim.append(0)
            list_range_total_dosage_units.append(0)
            list_range_total_spending.append(0)
            
            
    df_piv_merge_generic_risk['dose_price_range'] = pd.Series(list_max_min)
    df_piv_merge_generic_risk['total_bene_range'] = pd.Series(list_range_total_bene)
    df_piv_merge_generic_risk['total_claim_range'] = pd.Series(list_range_total_claim)
    df_piv_merge_generic_risk['total_dosage_range'] = pd.Series(list_range_total_dosage_units)
    df_piv_merge_generic_risk['total_spending_range'] = pd.Series(list_range_total_spending)
    
    df_class_generic = pd.pivot_table(df_piv_merge_generic_risk, index = ['generic_name'],
                                      values = ['risk_class', 'risk_count', 'total_beneficiaries',
                                                'total_claims', 'total_dosage_units',
                                                'total_spending', 'p_val_chisq', 
                                                'dose_price_range',
                                                'total_bene_range', 'total_claim_range',
                                                'total_dosage_range', 'total_spending_range',
                                                'serious_count', 'serious_count_pre',
                                                'serious_count_pre_pre', 'serious_range', 'chi_sq_val'],
                                      aggfunc = np.sum)
    df_class_generic_mean = pd.pivot_table(df_piv_merge_generic_risk, index = ['generic_name'],
                                      values = ['percent_change_spending', 'percent_change_dosage', 'percent_change_claims', 
                                                'percent_change_bene', 'percent_change_avg_spending_per_dose'],
                                      aggfunc = np.sum)

    df_class_generic = df_class_generic.merge(df_class_generic_mean, right_index=True,
                                              left_index=True, how='inner')
    classify_risk = df_class_generic['p_val_chisq']
    classify_risk[classify_risk == 0]= 1
    # For Holm Bonferroni correction
#    classify_risk = classify_risk.sort_values()
#    classify_risk_p_corr = []
#    for i in range(len(classify_risk),0,-1):
#        print(i)
#        classify_risk_p_corr.append(classify_risk[len(classify_risk)-i])
        
    p_val_cutoff = 0.05
    classify_risk[(classify_risk>=p_val_cutoff) | (classify_risk ==0)]=1
    classify_risk[classify_risk<p_val_cutoff]=0

    df_class_generic['classify_risk'] = classify_risk
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

def find_spending_change(df_spending_pre, df_spending_post):
    series_diff_spending = (df_spending_post[['total_spending']] - df_spending_pre[['total_spending']])/df_spending_pre[['total_spending']]
    series_diff_dosage = (df_spending_post[['total_dosage_units']] - df_spending_pre[['total_dosage_units']])/df_spending_pre[['total_dosage_units']]
    series_diff_claims = (df_spending_post[['total_claims']] - df_spending_pre[['total_claims']])/df_spending_pre[['total_claims']]
    series_diff_bene = (df_spending_post[['total_beneficiaries']] - df_spending_pre[['total_beneficiaries']])/df_spending_pre[['total_beneficiaries']]
    series_diff_avg_spending_per_dos = (df_spending_post[['average_spending_per_dosage_unit']] - df_spending_pre[['average_spending_per_dosage_unit']])/df_spending_pre[['average_spending_per_dosage_unit']]
    
    series_diff_spending.replace([np.inf, -np.inf], np.nan)
    series_diff_dosage.replace([np.inf, -np.inf], np.nan)
    series_diff_claims.replace([np.inf, -np.inf], np.nan)
    series_diff_bene.replace([np.inf, -np.inf], np.nan)
    series_diff_avg_spending_per_dos.replace([np.inf, -np.inf], np.nan)

    
    df_spending_difference = pd.concat([series_diff_spending, series_diff_dosage, series_diff_claims,
                                        series_diff_bene, series_diff_avg_spending_per_dos], axis=1)

    df_spending_difference.columns = ['percent_change_spending', 'percent_change_dosage', 
                                      'percent_change_claims', 'percent_change_bene',
                                      'percent_change_avg_spending_per_dose']
    return df_spending_difference

def merge_spending_diff(df_spending_current, df_spending_2013, df_spending_difference):
    # Merge spending by brand and adverse events
    df_spending_2013 = df_spending_2013.drop(['average_spending_per_claim', 
                                              'average_spending_per_beneficiary',
                                              'manufacturer'], axis=1)
    df_spending_tot = df_spending_2013.merge(df_spending_difference,
                                                  left_index =True,
                                                  right_index =True,
                                                  how = 'inner')
    df_spending_current.rename(columns={'average_spending_per_dosage_unit': 'current_avg_spending_per_dose'}, inplace=True)
    df_spending_tot['current_avg_spending_per_dose'] = df_spending_current['current_avg_spending_per_dose']
    return df_spending_tot


