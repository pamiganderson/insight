#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:41:24 2018

@author: pamelaanderson
"""

# Import needed libraries
import numpy as np
import pandas as pd
import time

from functions_adverse_drug_events import merge_ad_ev_tables, merge_2_me_tables
from functions_medicare_drug_costs import (read_spending_csv, format_str_and_numerics,
                                           create_ad_df_by_brand, merge_spending_df_and_ad_df,
                                           classify_generic_risk, feature_and_generic_label)
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
year = '2014'
df_spending = read_spending_csv(path, file_name, year)
df_spending = df_spending.reset_index(drop=True)
df_spending = format_str_and_numerics(df_spending)

########## OVER 2 MANUF ##########
## Find which drugs have >2 manufacturers
df_spending_2manuf = pd.pivot_table(df_spending, index = 'generic_name',
                                         values = 'manufacturer',
                                         aggfunc = 'count')
df_spending_2manuf = df_spending_2manuf[df_spending_2manuf['manufacturer'] > 1]
# Merge spending by brand and adverse events
df_spending = df_spending.drop(['manufacturer'], axis=1)
df_spending_adv_2manf = df_spending.merge(df_spending_2manuf,
                                           left_on = 'generic_name',
                                           right_index=True,
                                           how = 'inner')

########## MERGE SPENDING WITH AD EV BY BRAND ##########
# Merge spending with over 2 manufacturers with adverse event by brand
df_merge_ad_spending = merge_spending_df_and_ad_df(df_spending_adv_2manf,
                                                   df_ad_brand)

df_features_generic_class = feature_and_generic_label(df_merge_ad_spending)

# Classify generic vs. brand
df_merge_class = classify_generic_risk(df_features_generic_class)

df_merge_class_2 = df_merge_class[df_merge_class['risk_class'] == 0]
# Create Model
compare_classifiers(df_merge_class[['total_beneficiaries', 'total_claims',
                                    'total_dosage_units','total_spending',
                                    'manufacturer', 'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range']],df_merge_class[['classify_risk']])

# Random Forest Model
df_features = df_merge_class_2[['total_beneficiaries', 'total_claims',
                                    'total_spending',
                                    'manufacturer', 'dose_price_range',
                                    'total_bene_range', 'total_claim_range',
                                    'total_dosage_range', 'total_spending_range']]
resp_var = df_merge_class_2[['classify_risk']]
random_forest_model()

# Important features


# Chi sq test - contingency tables
from scipy.stats import chi2_contingency
obs = np.array([[60, 293329], [65, 65]])
chi2, p, dof, expected = chi2_contingency(obs)



