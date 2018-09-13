#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:13:29 2018

@author: pamelaanderson
"""
import numpy as np
import pandas as pd
import os
import json

def load_adverse_events(path, years):
    for year in years:
        path_w_year = path + year + '/'
        json_files = os.listdir(path_w_year)
        df_adverse_ev = pd.DataFrame()
        file_tot = [file for file in json_files if file not in ['.DS_Store']]
        ind = 0
        for file in file_tot:
            adverse_ev_data = json.load(open(path_w_year + file))
            df_adverse_ev_json = pd.DataFrame(adverse_ev_data['results'])
            df_adverse_ev = pd.concat([df_adverse_ev, df_adverse_ev_json])
            del adverse_ev_data, df_adverse_ev_json
            ind += 1
    df_adverse_ev = format_kept_cells(df_adverse_ev)
    df_adverse_ev = extract_drug_app_num_from_ad_ev(df_adverse_ev)
    df_adverse_ev = extract_patient_features(df_adverse_ev)
    df_adverse_ev = extract_drug_features(df_adverse_ev)
    df_adverse_ev = drop_unneeded_cols(df_adverse_ev)
    return df_adverse_ev

def drop_unneeded_cols(df_adverse_ev):
    drop_cols = ['authoritynumb','companynumb','duplicate', 'occurcountry',
                 'patient', 
                 'primarysource', 'primarysourcecountry', 'receiptdateformat',
                 'receivedate', 'receivedateformat', 'reportduplicate',
                 'reporttype','safetyreportid',
                 'safetyreportversion', 'sender',
                 'transmissiondate','transmissiondateformat']
    df_adverse_ev = df_adverse_ev.drop(drop_cols, axis=1)
    return df_adverse_ev

def format_kept_cells(df_adverse_ev):
    df_adverse_ev['fulfillexpeditecriteria'] = pd.to_numeric(df_adverse_ev['fulfillexpeditecriteria'])
    df_adverse_ev['serious'] = pd.to_numeric(df_adverse_ev['serious'])
    df_adverse_ev['receiptdate'] = pd.to_datetime(df_adverse_ev['receiptdate'])
    df_adverse_ev['source'] = pd.to_numeric(df_adverse_ev['primarysource']['qualification'])
    cols_to_convert_na_to_0 = ['seriousnesscongenitalanomali',
                                'seriousnessdeath',
                                'seriousnessdisabling',
                                'seriousnesshospitalization',
                                'seriousnesslifethreatening',
                                'seriousnessother']
    df_adverse_ev[ cols_to_convert_na_to_0 ] = df_adverse_ev[ cols_to_convert_na_to_0 ].fillna(value=0)
    return df_adverse_ev

def extract_drug_features(df_adverse_ev):
    medic_product = []
    drug_indict = []
    drug_route = []
    drug_char = []
    for i in range(0,len(df_adverse_ev)):
        col_names = list(df_adverse_ev.iloc[i]['patient']['drug'][0].keys())
        if 'medicinalproduct' in col_names:
            medic_product.append(drug_dict['medicinalproduct'])
        else:
            medic_product.append('nan')
        if 'drugindication' in col_names:
            drug_indict.append(drug_dict['drugindication'])
        else:
            drug_indict.append('nan')
        if 'drugadministrationroute' in col_names:
            drug_route.append(drug_dict['drugadministrationroute'])
        else:
            drug_route.append('nan')
        if 'drugcharacterization' in col_names:
            drug_char.append(drug_dict['drugcharacterization'])
        else:
            drug_char.append('nan')                
    drug_info = pd.DataFrame({'medic_product' : medic_product,
                              'drug_indict' : drug_indict,
                              'drug_route' : drug_route,
                             'drug_char' : drug_char})
    df_adverse_ev = pd.concat([df_adverse_ev, drug_info], axis = 1)
    return df_adverse_ev


def extract_patient_features(df_adverse_ev):
    patient_sex = []
    patient_age = []
    patient_reaction = []
    for i in range(0,len(df_adverse_ev)):
        col_names = list(df_adverse_ev_data.iloc[i]['patient'].keys())
        if 'patientsex' in col_names:
            patient_sex.append(df_adverse_ev_data.iloc[i]['patient']['patientsex'])
        else:
            patient_sex.append('nan')
        if 'patientonsetage' in col_names:
            patient_age.append(df_adverse_ev_data.iloc[i]['patient']['patientonsetage'])
        else:
            patient_age.append('nan')
        if 'reaction' in col_names:
            reaction_dict = df_adverse_ev_data.iloc[i]['patient']['reaction']
            reaction_score = []
            for k in range(0, len(reaction_dict)):
                col_names_react_dict = list(reaction_dict[k].keys())
                if 'reactionoutcome' in col_names_react_dict:
                    reaction_score.append(pd.to_numeric(reaction_dict[k]['reactionoutcome']))
            patient_reaction.append(np.mean(reaction_score))
        else:
            patient_reaction.append('nan')
    patient_info = pd.DataFrame({'patient_sex' : patient_sex,
                                 'patient_age' : patient_age,
                                 'patient_reaction' : patient_reaction})
    return patient_info    

def extract_drug_app_num_from_ad_ev(df_adverse_ev_data):
    drug_app_num = []
    drug_brand_name = []
    drug_generic_name = []
    drug_manuf_name =[]
    for i in range(0,len(df_adverse_ev_data)):
        col_names = list(df_adverse_ev_data.iloc[i]['patient']['drug'][0].keys())
        if 'openfda' in col_names:
            col_openfda_names = list(df_adverse_ev_data.iloc[i]['patient']['drug'][0]['openfda'].keys())
            if 'application_number' in col_openfda_names:
                drug_app_num.append(df_adverse_ev_data.iloc[i]['patient']['drug'][0]['openfda']['application_number'][0])
            else:
                drug_app_num.append('nan')
            if 'brand_name' in col_openfda_names:
                drug_brand_name.append(df_adverse_ev_data.iloc[i]['patient']['drug'][0]['openfda']['brand_name'][0])
            else:
                drug_brand_name.append('nan')
            if 'generic_name' in col_openfda_names:
                drug_generic_name.append(df_adverse_ev_data.iloc[i]['patient']['drug'][0]['openfda']['generic_name'][0])
            else:
                drug_generic_name.append('nan')
            if 'manufacturer_name' in col_openfda_names:
                drug_manuf_name.append(df_adverse_ev_data.iloc[i]['patient']['drug'][0]['openfda']['manufacturer_name'][0])
            else:
                drug_manuf_name.append('nan')
        else:
            drug_app_num.append('nan')
            drug_brand_name.append('nan')
            drug_generic_name.append('nan')
            drug_manuf_name.append('nan')
    drug_info = pd.DataFrame({'drug_app_num' : drug_app_num,
                              'drug_brand_name' : drug_brand_name,
                              'drug_generic_name' : drug_generic_name,
                              'drug_manuf_name' : drug_manuf_name})
    return drug_info

def extract_drug_app_num_from_recall(df_recall_w_openfda):
    drug_app_num = []
    for i in range(0,len(df_recall_w_openfda)):
        col_names = list(df_recall_w_openfda.iloc[i]['openfda'])
        if 'application_number' in col_names:
            drug_app_num.append(df_recall_w_openfda.iloc[i]['openfda']['application_number'])
        else:
            drug_app_num.append('nan')
    return drug_app_num