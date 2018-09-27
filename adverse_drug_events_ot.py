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
                                           find_num_manuf_and_change, find_spending_change)
from functions_ml_models import compare_classifiers, random_forest_model
from functions_data_cleaning import clean_ad_ev_table
from create_database import query_db_adverse_events

# Find data frame of adverse events in 2014
df_ad = query_db_adverse_events()

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
df_merge_class = df_merge_class.merge(df_manuf_per_generic,
                                      right_index=True,
                                      left_index=True,
                                      how='inner')


df_merge_class = df_merge_class.fillna(0)

df_merge_class_2 = df_merge_class[df_merge_class['risk_class'] == 0]


p = np.isinf(df_merge_class).any()
# add this back in 'diff_avg_spending_per_dose'

# Create Model
compare_classifiers(df_merge_class[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range',
                                    'diff_spending', 'diff_dosage', 'diff_claims', 
                                    'diff_bene', 'increase_manuf', 'total_manuf']],df_merge_class[['classify_risk']])

# Random Forest Model
df_features = df_merge_class[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range',
                                    'diff_spending', 'diff_dosage', 'diff_claims', 
                                    'diff_bene']]
resp_var = df_merge_class[['serious_count']]
random_forest_model()

# Important features


# Chi sq test - contingency tables
from scipy.stats import chi2_contingency
obs = np.array([[60, 293329], [65, 65]])
chi2, p, dof, expected = chi2_contingency(obs)



