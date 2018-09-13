#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:41:24 2018

@author: pamelaanderson
"""

# Import needed libraries
import pandas as pd
import time
import urllib.request, json

from functions_adverse_drug_events import extract_drug_auth_num_from_ad_ev





# Load adverse event data
years = ['2014'] # make string list of years to include adverse data for
path = '/Users/pamelaanderson/Documents/Insight/fda_drug_recall/'
df_adverse_ev = load_adverse_events(path, years)

col_names = list(df_adverse_ev.columns.values)
ad_examples = df_adverse_ev.iloc[0]
patient = ad_examples['patient']

start = time.time()
drug_application_number = extract_drug_auth_num_from_ad_ev(df_adverse_ev_data)
end = time.time()
print("Time to load one year = ", end-start)
patient_ex = df_adverse_ev_data.iloc[0]['patient']

# Recall data 
recall_data = json.load(open(path + "drug-enforcement-0001-of-0001.json"))
recall_data_results = recall_data['results']
df_recall_data_results = pd.DataFrame(recall_data_results)
df_recall_data_results['report_date'] = pd.to_datetime(df_recall_data_results['report_date'])

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

