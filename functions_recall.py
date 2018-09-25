#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:29:56 2018

@author: pamelaanderson
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime

def setup_recall_data(path):
    df_recall = load_recall_data(path)
    min_date = datetime.strptime('2014-01-01', '%Y-%m-%d')
    max_date = datetime.strptime('2015-12-31', '%Y-%m-%d')
    df_recall = df_restrict_recall_date_range(df_recall, min_date, max_date)
    df_recall = recall_class_to_numeric(df_recall)
    df_recall = drop_recall_cols(df_recall)
    df_recall = recall_lower_case(df_recall)
    df_recall_gb = pd.pivot_table(df_recall, index = ['product_description', 
                                                      'recalling_firm',
                                                      'reason_for_recall',
                                                      'recall_initiation_date'])
    df_recall_gb = df_recall_gb.reset_index()
    return df_recall

def load_recall_data(path):
    recall_data = json.load(open(path + "drug-enforcement-0001-of-0001.json"))
    recall_data = recall_data['results']
    df_recall_data = pd.DataFrame(recall_data)
    df_recall_data['report_date'] = pd.to_datetime(df_recall_data['report_date'])
    df_recall_data['recall_initiation_date'] = pd.to_datetime(df_recall_data['recall_initiation_date'])
    return df_recall_data

def df_restrict_recall_date_range(df_recall, min_date, max_date):
    df_recall['report_date'] = pd.to_datetime(df_recall['report_date'])
    df_recall['recall_initiation_date'] = pd.to_datetime(df_recall['recall_initiation_date'])
    df_recall = df_recall[df_recall['recall_initiation_date'] > min_date]
    df_recall = df_recall[df_recall['recall_initiation_date'] < max_date]
    return df_recall

def recall_class_to_numeric(df_recall):
    classification_list = []
    dict_recall_class = {'Class I' : 1,
                         'Class II' : 2,
                         'Class III' : 3,
                         'Not Yet Classified' : np.nan}
    for i in range(0,len(df_recall)):
        classification_list.append(dict_recall_class[df_recall.iloc[i]['classification']])
    df_recall['classification'] = classification_list
    df_recall = df_recall[df_recall['classification']!=3]
    return df_recall

def drop_recall_cols(df_recall):
    cols_to_drop = ['address_1', 'address_2', 'center_classification_date',
                    'city', 'code_info', 'country', 'distribution_pattern', 
                    'event_id', 'initial_firm_notification', 'more_code_info',
                    'product_type', 'postal_code', 'recall_number', 'state',
                    'termination_date']
    df_recall = df_recall.drop(cols_to_drop, axis=1)
    return df_recall

def recall_lower_case(df_recall):
    df_recall['product_description'] = df_recall['product_description'].str.lower()
    df_recall['product_quantity'] = df_recall['product_quantity'].str.lower()
    df_recall['reason_for_recall'] = df_recall['reason_for_recall'].str.lower()
    df_recall['recalling_firm'] = df_recall['recalling_firm'].str.lower()
    return df_recall

def save_raw_data(df_recall):
    df_recall.to_pickle("/Users/pamelaanderson/Documents/Insight/fda_drug_recall/dataframes/df_recall.pkl")
