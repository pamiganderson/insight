#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:41:24 2018

@author: pamelaanderson
"""

# Import needed libraries
import pandas as pd
import time

from functions_adverse_drug_events import merge_ad_ev_tables, merge_2_me_tables
from functions_medicare_drug_costs import (read_spending_csv, format_str_and_numerics,
                                           create_ad_df_by_brand, merge_spending_df_and_ad_df,
                                           classify_generic_risk)
from functions_ml_models import compare_classifiers, random_forest_model

# Find data frame of adverse events in 2014
df_merge_1 = merge_ad_ev_tables(df_ad_data_q1, df_ad_data_q2)
df_merge_2 = merge_ad_ev_tables(df_ad_data_q3, df_ad_data_q4)
df_ad_merge_2014 = merge_2_me_tables(df_merge_1 , df_merge_2)

# Clean adverse events table
clean_ad_ev_table(df_ad_merge_2014)

df_ad_merge_brand_2014 = create_ad_df_by_brand(df_ad_merge_2014)



# Find 2014 medicare spending by brand name
path = '/Users/pamelaanderson/Documents/Insight/spending/'
file_name = 'medicare_part_d_drug_spending.csv'
year = '2014'
df_spending_2014 = read_spending_csv(path, file_name, year)
df_spending_2014 = format_str_and_numerics(df_spending_2014)
df_merge_ad_spending = merge_spending_df_and_ad_df(df_spending_2014,
                                                   df_ad_merge_brand_2014)

# Find which drugs have >2 manufacturers
df_spending_2014_2manuf = pd.pivot_table(df_spending_2014, index = 'brand_name',
                                         values = 'manufacturer',
                                         aggfunc = 'count')
df_spending_2014_2manuf = df_spending_2014_2manuf[df_spending_2014_2manuf['manufacturer'] > 1]

# Merge spending by brand and adverse events
df_spending_adv_2manf = df_merge_ad_spending.merge(df_spending_2014_2manuf,
                                                   left_on = 'brand_name',
                                                   right_index=True,
                                                   how = 'inner')

df_spending_adv_2manf_nan = df_spending_adv_2manf.dropna()

# Classify generic vs. brand
df_merge_2014_class = classify_generic_risk(df_merge_2014)

# Create Model
compare_classifiers(df_class_generic[['sum_tot_bene',
                                         'sum_tot_claim',
                                         'sum_tot_dosage',
                                         'sum_tot_spend']], df_class_generic[['classify_risk']])

# Random Forest Model
random_forest_model()

# Important features


# Chi sq test - contingency tables
from scipy.stats import chi2_contingency
obs = np.array([[60, 293329], [65, 65]])
chi2, p, dof, expected = chi2_contingency(obs)

years = ['2014'] # make string list of years to include adverse data for
path = '/Users/pamelaanderson/Documents/Insight/fda_drug_recall/'
q_list = ['Q1', 'Q2', 'Q3', 'Q4']
i = 3
q = q_list[i]
df_adverse_ev = load_adverse_events(path, years, q_list[i])

col_names = list(df_adverse_ev.columns.values)
ad_examples = df_adverse_ev.iloc[0]
patient = ad_examples['patient']

start = time.time()
drug_application_number = extract_drug_auth_num_from_ad_ev(df_adverse_ev_data)
end = time.time()
print("Time to load one year = ", end-start)
patient_ex = df_adverse_ev_data.iloc[0]['patient']

# Recall data 


# If recall appplication number occurs within adverse app number

for j in range(0,len(drug_app_num)):
    if drug_app_num[j] is not 'nan':
        recall_ind = [i for i, x in enumerate(drug_auth_num) if x == drug_app_num[j]]
        if recall_ind != []:
            break

example = df_recall_data_results.iloc[1]
ex_manuf = example['openfda']

manuf_series_recall = df_recall_data_results['openfda'] 
tot_non_empty = 0
index_non_empty_openfda = []
for i in range(0,len(manuf_series_recall)):
    if manuf_series_recall[i]:
        tot_non_empty = tot_non_empty + 1
        index_non_empty_openfda.append(i)
        

# Determine the match between manufacturer_name and "recalling firm"
df_recall_w_openfda = df_recall_data_results.iloc[index_non_empty_openfda].reset_index()
match_manf_name_recall_firm = []
match_ratio_sim = []
import re
from difflib import SequenceMatcher
import numpy as np

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
# remove commas
# remove Inc
# remove .
# remove corp
# remove LLC
# remove USA
chars_to_remove = [',', '.', 'Inc', 'corp', 'LLC', 'USA']
rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

for i in range(0,len(df_recall_w_openfda)):
    recall_firm_name = df_recall_w_openfda.iloc[i]['recalling_firm']
    #recall_firm_name = re.sub(rx, '', recall_firm_name)
    manufac_name = df_recall_w_openfda.iloc[i]['openfda']['manufacturer_name'][0]
    match_ratio_sim.append(SequenceMatcher(None, recall_firm_name, manufac_name).ratio())
    if recall_firm_name in df_recall_w_openfda.iloc[i]['openfda']['manufacturer_name'][0]:
        match_manf_name_recall_firm.append(i)
num_matches_manufac = len(match_manf_name_recall_firm)
mean_match_ratio = np.mean(match_ratio_sim)

# Determine the match between recall: brand_name and recall:product description

# Determining classification and recall firm counts from total recall data
class_counts = df_recall_data_results['classification'].value_counts()
recalling_firm_counts = df_recall_data_results['recalling_firm'].value_counts().reset_index()

# Determining classification and recall firm counts from subset recall data
class_counts_openfda = df_recall_w_openfda['classification'].value_counts().reset_index()
recalling_firm_counts_openfda = df_recall_w_openfda['recalling_firm'].value_counts().reset_index()

pivot_classification = pd.pivot_table(df_recall_data_results, index = '')

