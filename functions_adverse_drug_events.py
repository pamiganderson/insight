#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:13:29 2018

@author: pamelaanderson
"""
from difflib import SequenceMatcher
import json
import numpy as np
import os
import operator
import pandas as pd



def load_adverse_events(path, year, q):
    """ Loading adverse drug events while performing basic pre-processing"""
    path_w_year = path + year + '/' + q + '/'
    json_files = os.listdir(path_w_year)
    df_adverse_ev = pd.DataFrame()
    file_tot = [file for file in json_files if file not in ['.DS_Store']]
    ind = 0
    for file in file_tot:
        print(file)
        adverse_ev_data = json.load(open(path_w_year + file))
        df_adverse_ev_json = pd.DataFrame(adverse_ev_data['results'])
        df_adverse_ev = pd.concat([df_adverse_ev, df_adverse_ev_json])
        del adverse_ev_data, df_adverse_ev_json
        ind += 1
    df_adverse_ev = df_adverse_ev.reset_index(drop=True)
    # Change data types to correct format
    df_adverse_ev = format_kept_cells(df_adverse_ev)
    # Find drug application number from nested dictionary
    df_adverse_ev = extract_drug_app_num_from_ad_ev(df_adverse_ev)
    # Find patient features from nested dictionary
    df_adverse_ev = extract_patient_features(df_adverse_ev)
    # Find drug info from nested dictionary
    df_adverse_ev = extract_drug_features(df_adverse_ev)
    # Find who submitted report info as column in df
    df_adverse_ev = extract_source_info(df_adverse_ev)
    # Drop columns that will not be included as features
    df_adverse_ev = drop_unneeded_cols(df_adverse_ev)
    return df_adverse_ev


def drop_unneeded_cols(df_adverse_ev):
    """ Drop the columns that will not be used as features """
    drop_cols = ['companynumb','duplicate', 'occurcountry',
                 'patient', 
                 'primarysourcecountry', 'receiptdateformat',
                 'receiver', 'receivedate', 'receivedateformat', 'reportduplicate',
                 'reporttype','safetyreportid',
                 'safetyreportversion', 'sender',
                 'transmissiondate','transmissiondateformat']
    df_adverse_ev = df_adverse_ev.drop(drop_cols, axis=1)
    return df_adverse_ev

def format_kept_cells(df_adverse_ev):
    """ Correct data types (to numeric or datetime) """
    df_adverse_ev['fulfillexpeditecriteria'] = pd.to_numeric(df_adverse_ev['fulfillexpeditecriteria'])
    df_adverse_ev['serious'] = pd.to_numeric(df_adverse_ev['serious'])
    df_adverse_ev['seriousnesscongenitalanomali'] = pd.to_numeric(df_adverse_ev['seriousnesscongenitalanomali'])
    df_adverse_ev['seriousnessdeath'] = pd.to_numeric(df_adverse_ev['seriousnessdeath'])
    df_adverse_ev['seriousnessdisabling'] = pd.to_numeric(df_adverse_ev['seriousnessdisabling'])
    df_adverse_ev['seriousnesshospitalization'] = pd.to_numeric(df_adverse_ev['seriousnesshospitalization'])
    df_adverse_ev['seriousnesslifethreatening'] = pd.to_numeric(df_adverse_ev['seriousnesslifethreatening'])
    df_adverse_ev['seriousnessother'] = pd.to_numeric(df_adverse_ev['seriousnessother'])
    df_adverse_ev['receiptdate'] = pd.to_datetime(df_adverse_ev['receiptdate'])
    cols_to_convert_na_to_0 = ['serious',
                               'seriousnesscongenitalanomali',
                               'seriousnessdeath',
                               'seriousnessdisabling',
                               'seriousnesshospitalization',
                               'seriousnesslifethreatening',
                               'seriousnessother']
    df_adverse_ev[cols_to_convert_na_to_0] = df_adverse_ev[ cols_to_convert_na_to_0 ].fillna(value=0)
    return df_adverse_ev


def extract_drug_features(df_adverse_ev):
    """ Find the relevant information about the drugs """
    medic_product = []
    drug_indict = []
    drug_route = []
    drug_char = []
    for i in range(0,len(df_adverse_ev)):
        col_names = list(df_adverse_ev.iloc[i]['patient']['drug'][0].keys())
        if 'medicinalproduct' in col_names:
            medic_product.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['medicinalproduct'])
        else:
            medic_product.append(np.nan)
        if 'drugindication' in col_names:
            drug_indict.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['drugindication'])
        else:
            drug_indict.append(np.nan)
        if 'drugadministrationroute' in col_names:
            drug_route.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['drugadministrationroute'])
        else:
            drug_route.append(np.nan)
        if 'drugcharacterization' in col_names:
            drug_char.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['drugcharacterization'])
        else:
            drug_char.append(np.nan)                
    drug_info = pd.DataFrame({'medic_product' : medic_product,
                              'drug_indict' : drug_indict,
                              'drug_route' : drug_route,
                             'drug_char' : drug_char})
    df_adverse_ev = pd.concat([df_adverse_ev, drug_info], axis=1)
    return df_adverse_ev

def extract_source_info(df_adverse_ev):
    """ Find information about who submitted the report """
    qual_list = []
    for i in range(0,len(df_adverse_ev)):
        if df_adverse_ev.iloc[i]['primarysource'] is not None:
            col_names = list(df_adverse_ev.iloc[i]['primarysource'].keys())
            if 'qualification' in col_names:
                qual_list.append(pd.to_numeric(df_adverse_ev.iloc[i]['primarysource']['qualification']))
            else:
                qual_list.append(np.nan)
        else:
            qual_list.append(np.nan)
    df_adverse_ev['source'] = qual_list
    df_adverse_ev = df_adverse_ev.drop(['primarysource'], axis=1)
    return df_adverse_ev
    

def extract_patient_features(df_adverse_ev):
    """ Find information about the patient with the ADR """
    patient_sex = []
    patient_age = []
    patient_reaction = []
    patient_reaction_type = []
    for i in range(0,len(df_adverse_ev)):
        col_names = list(df_adverse_ev.iloc[i]['patient'].keys())
        if 'patientsex' in col_names:
            patient_sex.append(pd.to_numeric(df_adverse_ev.iloc[i]['patient']['patientsex']))
        else:
            patient_sex.append(np.nan)
        if 'patientonsetage' in col_names:
            patient_age.append(pd.to_numeric(df_adverse_ev.iloc[i]['patient']['patientonsetage']))
        else:
            patient_age.append(np.nan)
        if 'reaction' in col_names:
            reaction_dict = df_adverse_ev.iloc[i]['patient']['reaction']
            reaction_score = []
            reaction_type = []
            for k in range(0, len(reaction_dict)):
                col_names_react_dict = list(reaction_dict[k].keys())
                if 'reactionoutcome' in col_names_react_dict:
                    reaction_score.append(pd.to_numeric(reaction_dict[k]['reactionoutcome']))
                if 'reactionmeddrapt' in col_names_react_dict:
                    reaction_type.append(reaction_dict[k]['reactionmeddrapt'])
            patient_reaction_type.append(reaction_type[0])
            patient_reaction.append(np.mean(reaction_score))
        else:
            patient_reaction.append(np.nan)
            patient_reaction_type.append(np.nan)
    patient_info = pd.DataFrame({'patient_sex' : patient_sex,
                                 'patient_age' : patient_age,
                                 'patient_reaction' : patient_reaction,
                                 'patient_react_type' : patient_reaction_type})
    df_adverse_ev = pd.concat([df_adverse_ev, patient_info], axis= 1)
    return df_adverse_ev


def extract_drug_app_num_from_ad_ev(df_adverse_ev):
    """ Find the brand name, generic name, and manuf name for each drug """
    drug_app_num = []
    drug_brand_name = []
    drug_generic_name = []
    drug_manuf_name =[]
    for i in range(0,len(df_adverse_ev)):
        col_names = list(df_adverse_ev.iloc[i]['patient']['drug'][0].keys())
        if 'openfda' in col_names:
            col_openfda_names = list(df_adverse_ev.iloc[i]['patient']['drug'][0]['openfda'].keys())
            if 'application_number' in col_openfda_names:
                drug_app_num.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['openfda']['application_number'][0])
            else:
                drug_app_num.append(np.nan)
            if 'brand_name' in col_openfda_names:
                drug_brand_name.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['openfda']['brand_name'][0])
            else:
                drug_brand_name.append(np.nan)
            if 'generic_name' in col_openfda_names:
                drug_generic_name.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['openfda']['generic_name'][0])
            else:
                drug_generic_name.append(np.nan)
            if 'manufacturer_name' in col_openfda_names:
                drug_manuf_name.append(df_adverse_ev.iloc[i]['patient']['drug'][0]['openfda']['manufacturer_name'][0])
            else:
                drug_manuf_name.append(np.nan)
        else:
            drug_app_num.append(np.nan)
            drug_brand_name.append(np.nan)
            drug_generic_name.append(np.nan)
            drug_manuf_name.append(np.nan)
    drug_info = pd.DataFrame({'drug_app_num' : drug_app_num,
                              'drug_brand_name' : drug_brand_name,
                              'drug_generic_name' : drug_generic_name,
                              'drug_manuf_name' : drug_manuf_name})
    df_adverse_ev = pd.concat([df_adverse_ev, drug_info], axis=1)
    return df_adverse_ev


def extract_drug_app_num_from_recall(df_recall_w_openfda):
    """ Find drug application number """
    drug_app_num = []
    for i in range(0,len(df_recall_w_openfda)):
        col_names = list(df_recall_w_openfda.iloc[i]['openfda'])
        if 'application_number' in col_names:
            drug_app_num.append(df_recall_w_openfda.iloc[i]['openfda']['application_number'])
        else:
            drug_app_num.append(np.nan)
    return drug_app_num    


def create_ad_ev_pivot_on_drug_num(df_adverse_ev):
    df_ad_ev_drug_num = pd.pivot_table(df_adverse_ev, index = ['drug_generic_name',
                                                               'drug_brand_name',
                                                               'drug_manuf_name']) 
    df_ad_ev_drug_num = df_ad_ev_drug_num.reset_index()
    return df_ad_ev_drug_num


def merge_ad_ev_tables(df_ad_data_q1, df_ad_data_q2):
    """ Merge adverse event df from different quarters together """
    cols_drop_indiv = ['seriousnesscongenitalanomali_count',
                 'seriousnessdeath_count',
                 'seriousnessdisabling_count',
                 'seriousnesshospitalization_count',
                 'seriousnesslifethreatening_count',
                 'seriousnessother_count',
                 'age_count']
    cols_drop = ['serious_count_x',
                 'serious_count_y',
                 'drug_generic_name_x',
                 'drug_generic_name_y',
                 'drug_manuf_name_x',
                 'drug_manuf_name_y']
    df_ad_data_q1 = df_ad_data_q1.drop(cols_drop_indiv, axis=1)
    df_ad_data_q2 = df_ad_data_q2.drop(cols_drop_indiv, axis=1)
    df_ad_data_merge = df_ad_data_q1.merge(df_ad_data_q2, on = 'drug_brand_name',
                                           how = 'outer')
    df_ad_data_merge = df_ad_data_merge.fillna(0)
    df_ad_data_merge['serious_count'] = (df_ad_data_merge['serious_count_x'] +
                                            df_ad_data_merge['serious_count_y'])
    drug_gen_list = []
    drug_manuf_list = []
    for i in range(0, len(df_ad_data_merge)):
        if df_ad_data_merge.iloc[i]['drug_generic_name_x'] == 0:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_y'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_y'])
        else:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_x'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_x'])
    df_ad_data_merge['drug_generic_name'] = pd.Series(drug_gen_list)
    df_ad_data_merge['drug_manuf_name'] = pd.Series(drug_manuf_list)
    df_ad_data_merge = df_ad_data_merge.drop(cols_drop, axis=1)
    return df_ad_data_merge


def merge_ad_ev_tables_serious(df_ad_data_q1, df_ad_data_q2):
    """ Merge two ad events df and sum the serious events """
    cols_drop = ['serious_count_x',
                 'serious_count_y',
                 'drug_generic_name_x',
                 'drug_generic_name_y',
                 'drug_manuf_name_x',
                 'drug_manuf_name_y']
    df_ad_data_merge = df_ad_data_q1.merge(df_ad_data_q2, on = 'drug_brand_name',
                                           how = 'outer')
    df_ad_data_merge = df_ad_data_merge.fillna(0)
    df_ad_data_merge['serious_count'] = (df_ad_data_merge['serious_count_x'] +
                                            df_ad_data_merge['serious_count_y'])
    drug_gen_list = []
    drug_manuf_list = []
    for i in range(0, len(df_ad_data_merge)):
        if df_ad_data_merge.iloc[i]['drug_generic_name_x'] == 0:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_y'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_y'])
        else:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_x'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_x'])
    df_ad_data_merge['drug_generic_name'] = pd.Series(drug_gen_list)
    df_ad_data_merge['drug_manuf_name'] = pd.Series(drug_manuf_list)
    df_ad_data_merge = df_ad_data_merge.drop(cols_drop, axis=1)
    return df_ad_data_merge


def merge_2_me_tables_serious(df_ad_data_merge_1, df_ad_data_merge_2):
    """ Merge the two complete adverse event data frames """
    cols_drop = ['serious_count_x',
                 'serious_count_y',
                 'drug_generic_name_x',
                 'drug_generic_name_y',
                 'drug_manuf_name_x',
                 'drug_manuf_name_y']
    df_ad_data_merge =df_ad_data_merge_1.merge(df_ad_data_merge_2, on = 'drug_brand_name',
                                               how = 'outer')
    df_ad_data_merge = df_ad_data_merge.fillna(0)
    df_ad_data_merge['serious_count'] = (df_ad_data_merge['serious_count_x'] +
                                        df_ad_data_merge['serious_count_y'])
    drug_gen_list = []
    drug_manuf_list = []
    for i in range(0, len(df_ad_data_merge)):
        if df_ad_data_merge.iloc[i]['drug_generic_name_x'] == 0:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_y'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_y'])
        else:
            drug_gen_list.append(df_ad_data_merge.iloc[i]['drug_generic_name_x'])
            drug_manuf_list.append(df_ad_data_merge.iloc[i]['drug_manuf_name_x'])
 
    df_ad_data_merge['drug_generic_name'] = pd.Series(drug_gen_list)
    df_ad_data_merge['drug_manuf_name'] = pd.Series(drug_manuf_list)
    df_ad_data_merge = df_ad_data_merge.drop(cols_drop, axis=1)
    return df_ad_data_merge


def classify_ad_event_recall(df_ad_ev_drug_num, df_recall):
    """ Classify the adverse event data corresponding to drug recall"""
    df_ad_ev_drug_num = df_ad_ev_drug_num.fillna('MISSING')
    df_ad_ev_drug_num['drug_generic_name'] = df_ad_ev_drug_num['drug_generic_name'].str.lower()
    df_ad_ev_drug_num['drug_brand_name'] = df_ad_ev_drug_num['drug_brand_name'].str.lower()
    df_ad_ev_drug_num['drug_manuf_name'] = df_ad_ev_drug_num['drug_manuf_name'].str.lower()
    recall_sim_manuf_list = []
    recall_bool_list = []
    recall_class_list = []
    recall_reason_list = []
    recall_report_date_list = []
    for i in range(0,len(df_ad_ev_drug_num)):
        bool_series_gen = df_recall['product_description'].str.contains(df_ad_ev_drug_num.iloc[i]['drug_generic_name'])
        bool_series_brand = df_recall['product_description'].str.contains(df_ad_ev_drug_num.iloc[i]['drug_brand_name'])
        bool_series = (bool_series_gen | bool_series_brand)
        if np.sum(bool_series) > 0:
            df_recall_sub = df_recall[bool_series].reset_index()
            manuf_sim_list = []
            for k in range(0, len(df_recall_sub)):
                manuf_sim_list.append(SequenceMatcher(None, df_recall_sub.iloc[k]['recalling_firm'],
                                        df_ad_ev_drug_num.iloc[i]['drug_manuf_name']).ratio())
            max_ind, max_val = max(enumerate(manuf_sim_list), key=operator.itemgetter(1))
            recall_sim_manuf_list.append(max_val)
            if max_val > 0.9:
                recall_bool_list.append(1)
                recall_class_list.append(df_recall_sub.iloc[max_ind]['classification'])
                recall_reason_list.append(df_recall_sub.iloc[max_ind]['reason_for_recall'])
                recall_report_date_list.append(df_recall_sub.iloc[max_ind]['recall_initiation_date'])
            else:
                recall_bool_list.append(0)
                recall_class_list.append(np.nan)
                recall_reason_list.append(np.nan)
                recall_report_date_list.append(np.nan)
        else:
            recall_sim_manuf_list.append(np.nan)
            recall_bool_list.append(0)
            recall_class_list.append(np.nan)
            recall_reason_list.append(np.nan)
            recall_report_date_list.append(np.nan)
    df_recall_info = pd.DataFrame({'recall_bool' : recall_bool_list,
                                   'recall_sim_manuf' : recall_sim_manuf_list,
                                   'recall_class' : recall_class_list,
                                   'recall_reason' : recall_reason_list,
                                   'recall_report_date' : recall_report_date_list})
    df_ad_ev_drug_num = pd.concat([df_ad_ev_drug_num, df_recall_info], axis=1)
    
    
def save_raw_data(df_adverse_ev):
    """ Pickle the adverse event dataframe to save and load later"""
    df_adverse_ev.to_pickle("/Users/pamelaanderson/Documents/Insight/fda_drug_recall/dataframes/df_adverse_ev_2014_q1.pkl")
    ## Data to pickle
    #df_merge_ad_spending_label_save = df_merge_ad_spending_label
    #df_merge_ad_spending_label_save = df_merge_ad_spending_label_save.drop(warfarin_drop)
    df_merge_ad_spending_label_save.to_pickle('./data/df_merge_ad_spending.pkl')
    
    
    df_merge_classify_final = df_merge_classify_final.drop([0,1])
    df_merge_classify_final.to_pickle('./data/df_merge_classify_final.pkl')
    
    
    df_patient_react = df_serious_clean_brand
    df_serious_clean_brand.to_pickle('./df_patient_react.pkl')