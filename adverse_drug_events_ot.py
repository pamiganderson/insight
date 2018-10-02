#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:09:41 2018

@author: pamelaanderson
"""

# Import needed libraries
import numpy as np
import pandas as pd
import time

from functions_adverse_drug_events import merge_ad_ev_tables, merge_2_me_tables
from functions_medicare_drug_costs import (read_spending_csv, format_str_and_numerics,
                                           create_ad_df_by_brand, merge_spending_df_tot_and_ad_df,
                                           classify_generic_risk, feature_and_generic_label,
                                           find_num_manuf_and_change, find_spending_change,
                                           merge_spending_diff)
from functions_ml_models import compare_classifiers, random_forest_model
from functions_data_cleaning import clean_ad_ev_table
from functions_drug_features import find_num_act_ingredients, find_nti_drugs
from create_database import query_db_adverse_events

# Find data frame of adverse events in 2014
df_ad = query_db_adverse_events()
df_ad['serious_count'] = df_ad['serious_count']

# Clean adverse events table
df_ad_clean = clean_ad_ev_table(df_ad)
df_ad_clean = df_ad_clean[['drug_generic_name_re', 'drug_brand_name_re', 'serious_count']]
df_ad_clean.columns = ['drug_generic_name', 'drug_brand_name', 'serious_count']
# Adverse events df by drug brand
df_ad_brand = create_ad_df_by_brand(df_ad_clean)

########## FIND 2014 MEDICARE SPENDING ##########
# Find 2014 medicare spending by brand name
path = '/Users/pamelaanderson/Documents/Insight/spending/'
file_name = 'medicare_part_d_drug_spending.csv'
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

df_spending_tot = merge_spending_diff(df_spending_2013, df_spending_difference)

########## MERGE SPENDING WITH AD EV BY BRAND ##########
# Merge spending with over 2 manufacturers with adverse event by brand
df_merge_ad_spending = merge_spending_df_tot_and_ad_df(df_spending_tot,
                                                       df_ad_brand)



df_features_generic_class = feature_and_generic_label(df_merge_ad_spending)

# Classify generic vs. brand
df_merge_class = classify_generic_risk(df_features_generic_class)

# Add drug features by generic type
df_merge_class = find_nti_drugs(df_merge_class)
df_merge_class = find_num_act_ingredients(df_merge_class)
df_merge_class = df_merge_class.merge(df_manuf_per_generic,
                                      right_index=True,
                                      left_index=True,
                                      how='inner')

#price_incr_decr = df_merge_class['diff_avg_spending_per_dose']
#price_incr_decr[price_incr_decr < 0] = 0
#price_incr_decr[price_incr_decr > 0] = 1
#df_merge_class['price_change'] = price_incr_decr

df_merge_class = df_merge_class.fillna(0)


#manuf = df_manuf_per_generic.copy()
#incre_manuf = manuf['increase_manuf']
#incre_manuf[incre_manuf < 0] = 0
#incre_manuf[incre_manuf > 0] = 1
#df_merge_class_t['increase_manuf'] = df_merge_class_t['increase_manuf']



p = np.isinf(df_merge_class).any()
df_merge_class['diff_avg_spending_per_dose'][np.isinf(df_merge_class['diff_avg_spending_per_dose'])] = 0
df_merge_class_2 = df_merge_class[(df_merge_class['risk_class'] == 0)]
df_merge_class_d = df_merge_class[(df_merge_class['total_beneficiaries'] > 100)]

# add this back in 'diff_avg_spending_per_dose'


# Create Model
compare_classifiers(df_merge_class[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range',
                                    'diff_spending', 'diff_dosage', 'diff_claims', 
                                    'diff_bene', 'increase_manuf', 'total_manuf',
                                    'diff_avg_spending_per_dose', 'nti_index',
                                    'num_act_ingredients']],df_merge_class[['classify_risk']])

# Random Forest Model
df_features = df_merge_class[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range',
                                    'diff_spending', 'diff_dosage', 'diff_claims', 
                                    'diff_bene', 'increase_manuf', 'total_manuf',
                                    'diff_avg_spending_per_dose', 'nti_index',
                                    'num_act_ingredients']]
resp_var = df_merge_class[['classify_risk']]
random_forest_model()

# Examining the data
#p = df_merge_ad_spending[['brand_name', 'generic_name', 'total_beneficiaries',
#                          'serious_count', 'serious_per_bene']]


# Important features


# Chi sq test - contingency tables
from scipy.stats import chi2_contingency
obs = np.array([[60, 293329], [65, 65]])
chi2, p, dof, expected = chi2_contingency(obs)



