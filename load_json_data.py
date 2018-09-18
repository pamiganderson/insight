#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 22:30:57 2018

@author: pamelaanderson
"""
import os
import pandas as pd
import urllib.request, json

def load_adverse_events(path, years):
    q_list = ['Q1', 'Q2', 'Q3', 'Q4']
    for year in years:
        for q in q_list:
            path_w_year = path + year + '/' + q + '/'
            json_files = os.listdir(path_w_year)
            df_adverse_ev = pd.DataFrame()
            file_tot = [file for file in json_files if file not in ['.DS_Store']]
            ind = 0
            for file in file_tot:
                adverse_ev_data = json.load(open(path_w_year + file))
                #df_adverse_ev_json = pd.read_json(path_w_year + file, lines=True, chunksize=10000)
                df_adverse_ev_json = pd.DataFrame(adverse_ev_data['results'])
                df_adverse_ev = pd.concat([df_adverse_ev, df_adverse_ev_json])
                del adverse_ev_data, df_adverse_ev_json
                ind += 1
    df_adverse_ev = df_adverse_ev.reset_index(drop=True)
    df_adverse_ev = format_kept_cells(df_adverse_ev)
    df_adverse_ev = extract_drug_app_num_from_ad_ev(df_adverse_ev)
    df_adverse_ev = extract_patient_features(df_adverse_ev)
    df_adverse_ev = extract_drug_features(df_adverse_ev)
    df_adverse_ev = extract_source_info(df_adverse_ev)
    df_adverse_ev = drop_unneeded_cols(df_adverse_ev)
    return df_adverse_ev