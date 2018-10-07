#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:09:41 2018

@author: pamelaanderson
"""

# Import needed libraries
import numpy as np
import pandas as pd

from functions_drug_features import find_num_act_ingredients, find_nti_drugs
from functions_load_tables import load_ad_ev_df
from functions_medicare_drug_costs import (read_spending_csv, format_str_and_numerics,
                                           create_ad_df_by_brand, merge_spending_df_tot_and_ad_df,
                                           classify_generic_risk, feature_and_generic_label,
                                           find_num_manuf_and_change, find_spending_change,
                                           merge_spending_diff, label_brand_generic)
from functions_ml_models import compare_classifiers, random_forest_model


# Load in adverse event tables
year_list = ['2012', '2013']
year_to_predict = '2014'
year_list.append(year_to_predict)
dict_df_adverse_ev = load_ad_ev_df(year_list)


# Adverse events df by drug brand
df_ad_brand = create_ad_df_by_brand(dict_df_adverse_ev['df_ad_clean'])
df_ad_brand_prev_1y = create_ad_df_by_brand(dict_df_adverse_ev['df_ad_clean_pre'])
df_ad_brand_prev_2y = create_ad_df_by_brand(dict_df_adverse_ev['df_ad_clean_pre_pre'])

# Create function
df_ad_brand_prev_merge = df_ad_brand_prev_1y.merge(df_ad_brand_prev_2y,
                                              left_on='drug_brand_name',
                                              right_on='drug_brand_name',
                                              how='outer')
drug_gen_list = []
for i in range(0, len(df_ad_brand_prev_merge)):
    if df_ad_brand_prev_merge.iloc[i]['drug_generic_name_x'] == 0:
        drug_gen_list.append(df_ad_brand_prev_merge.iloc[i]['drug_generic_name_y'])
    else:
        drug_gen_list.append(df_ad_brand_prev_merge.iloc[i]['drug_generic_name_x'])
df_ad_brand_prev_merge['drug_generic_name'] = pd.Series(drug_gen_list)
df_ad_brand_prev_merge = df_ad_brand_prev_merge.drop(['drug_generic_name_x', 
                                                    'drug_generic_name_y'], axis=1)
df_ad_brand_prev_merge.rename(columns={'serious_count_x': 'serious_count_pre',
                                      'serious_count_y':'serious_count_pre_pre'}, inplace=True)
df_ad_brand_prev_merge['serious_range'] = df_ad_brand_prev_merge['serious_count_pre'] - df_ad_brand_prev_merge['serious_count_pre_pre']



########## FIND 2014 MEDICARE SPENDING ##########
# Find 2014 medicare spending by brand name
path = '/Users/pamelaanderson/Documents/Insight/spending/'
file_name = 'medicare_part_d_drug_spending.csv'
year = '2014'
df_spending_2014 = read_spending_csv(path, file_name, year)
df_spending_2014 = df_spending_2014.reset_index(drop=True)
df_spending_2014 = format_str_and_numerics(df_spending_2014)
year = '2013'
df_spending_2013 = read_spending_csv(path, file_name, year)
df_spending_2013 = df_spending_2013.reset_index(drop=True)
df_spending_2013 = format_str_and_numerics(df_spending_2013)
year = '2012'
df_spending_2012 = read_spending_csv(path, file_name, year)
df_spending_2012 = df_spending_2012.reset_index(drop=True)
df_spending_2012 = format_str_and_numerics(df_spending_2012)

########## OVER 2 MANUF ##########
## Find which drugs have >2 manufacturers
df_manuf_per_generic = find_num_manuf_and_change(df_spending_2012, df_spending_2013)

df_spending_difference = find_spending_change(df_spending_2012, df_spending_2013)

df_spending_tot = merge_spending_diff(df_spending_2014, df_spending_2013, df_spending_difference)

########## MERGE SPENDING WITH AD EV BY BRAND ##########
# Merge spending with over 2 manufacturers with adverse event by brand
df_ad_brand_total = df_ad_brand.merge(df_ad_brand_prev_merge, left_on='drug_brand_name',
                                      right_on='drug_brand_name', how='outer')
df_ad_brand_total = df_ad_brand_total.fillna(0)
drug_gen_list = []
for i in range(0, len(df_ad_brand_total)):
    if df_ad_brand_total.iloc[i]['drug_generic_name_x'] == 0:
        drug_gen_list.append(df_ad_brand_total.iloc[i]['drug_generic_name_y'])
    else:
        drug_gen_list.append(df_ad_brand_total.iloc[i]['drug_generic_name_x'])
df_ad_brand_total['drug_generic_name'] = pd.Series(drug_gen_list)
df_ad_brand_total = df_ad_brand_total.drop(['drug_generic_name_x', 
                                            'drug_generic_name_y'], axis=1)

# Sumarize the spending information from CMS
df_merge_ad_spending = merge_spending_df_tot_and_ad_df(df_spending_tot,
                                                       df_ad_brand_total)
# Find generic/brand flags for each drug in the CMS database
df_merge_ad_spending_label = label_brand_generic(df_merge_ad_spending)

df_features_generic_class = feature_and_generic_label(df_merge_ad_spending_label)

# Classify generic vs. brand
df_merge_class = classify_generic_risk(df_features_generic_class)

# Add drug features by generic type
df_merge_class = find_nti_drugs(df_merge_class)
df_merge_class = find_num_act_ingredients(df_merge_class)
df_merge_class = df_merge_class.merge(df_manuf_per_generic,
                                      right_index=True,
                                      left_index=True,
                                      how='inner')
df_merge_class = df_merge_class.fillna(0)
p = np.isinf(df_merge_class).any()
df_merge_class['percent_change_avg_spending_per_dose'][np.isinf(df_merge_class['percent_change_avg_spending_per_dose'])] = 0
df_merge_class_2 = df_merge_class[(df_merge_class['risk_class'] == 0)]
df_merge_class_d = df_merge_class[(df_merge_class['total_beneficiaries'] > 100)]


# Create Model
results_plot = compare_classifiers(df_merge_class_2[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range',
                                    'percent_change_spending', 'percent_change_dosage', 'percent_change_claims', 
                                    'percent_change_bene', 'percent_change_avg_spending_per_dose',  
                                    'increase_manuf', 'total_manuf',
                                    'serious_count_pre',
                                    'serious_count_pre_pre', 'serious_range',
                                    'nti_index', 'num_act_ingredients']], df_merge_class_2[['classify_risk']])
# Random Forest Model
df_features = df_merge_class_2[['total_beneficiaries', 'total_claims',
                                'total_dosage_units',
                                'total_spending',
                                'dose_price_range',
                                'total_bene_range', 
                                'total_spending_range',
                                'percent_change_spending', 'percent_change_dosage',
                                'percent_change_claims', 
                                'percent_change_bene', 'percent_change_avg_spending_per_dose',
                                'increase_manuf', 'total_manuf',
                                'nti_index', 'num_act_ingredients',
                                'serious_count_pre',
                                'serious_count_pre_pre', 'serious_range']]
resp_var = df_merge_class_2[['classify_risk']]
#resp_var = df_merge_class_2[['chi_sq_val']]
random_forest_model()

# Excluding these features:
# 'total_claim_range','total_dosage_range',
# Examining the data
p = df_merge_ad_spending[['brand_name', 'generic_name', 'total_beneficiaries',
                          'serious_count', 'serious_per_bene']]


# Important features


# Chi sq test - contingency tables
from scipy.stats import chi2_contingency
obs = np.array([[60, 293329], [65, 65]])
chi2, p, dof, expected = chi2_contingency(obs)


# 
df_merge_classify_final = df_merge_class.reset_index()
df_merge_classify_final.rename(columns={'index': 'generic_name'}, inplace=True)

amiodarone_drop = [594]
warfarin_drop = [1304]

## Data to pickle
#df_merge_ad_spending_label_save = df_merge_ad_spending_label
#df_merge_ad_spending_label_save = df_merge_ad_spending_label_save.drop(warfarin_drop)
df_merge_ad_spending_label_save.to_pickle('./data/df_merge_ad_spending.pkl')


df_merge_classify_final = df_merge_classify_final.drop([0,1])
df_merge_classify_final.to_pickle('./data/df_merge_classify_final.pkl')


df_patient_react = df_serious_clean_brand
df_serious_clean_brand.to_pickle('./df_patient_react.pkl')


#price_incr_decr = df_merge_class['diff_avg_spending_per_dose']
#price_incr_decr[price_incr_decr < 0] = 0
#price_incr_decr[price_incr_decr > 0] = 1
#df_merge_class['price_change'] = price_incr_decr

#manuf = df_manuf_per_generic.copy()
#incre_manuf = manuf['increase_manuf']
#incre_manuf[incre_manuf < 0] = 0
#incre_manuf[incre_manuf > 0] = 1
#df_merge_class_t['increase_manuf'] = df_merge_class_t['increase_manuf']