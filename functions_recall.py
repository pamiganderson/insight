#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:29:56 2018

@author: pamelaanderson
"""
import pandas as pd
min_date = min(df_adverse_ev['receiptdate'])
max_date = max(df_adverse_ev['receiptdate'])

def df_restrict_recall_date_range(df_recall, min_date, max_date):
    df_recall['report_date'] = pd.to_datetime(df_recall['report_date'])
    df_recall = df_recall[df_recall['report_date'] > min_date]
    df_recall = df_recall[df_recall['report_date'] < max_date]
    return df_recall

def recall_class_to_numeric(df_recall):
    classification_list = []
    dict_recall_class = {'Class I' : 1,
                         'Class II' : 2,
                         'Class III' : 3}
    for i in range(0,len(df_recall)):
        classification_list.append(dict_recall_class[df_recall.iloc[i]['classification']])
    df_recall['classification'] = classification_list
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