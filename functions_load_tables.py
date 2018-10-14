#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:20:12 2018

@author: pamelaanderson
"""
import psycopg2
import pandas as pd


from functions_adverse_drug_events import (merge_ad_ev_tables_serious, 
                                           merge_2_me_tables_serious)
from functions_data_cleaning import clean_ad_ev_table




def load_ad_ev_df(year_list):
    dict_df_ad = create_ad_ev_df(year_list)
    dict_df_clean_ad = {}
    for i in list(dict_df_ad.keys()):
        if i == '2012':
            df_ad_2012_q1 = dict_df_ad[i]['df_adverse_ev_2012_q1_table']
            df_ad_2012_q2 = dict_df_ad[i]['df_adverse_ev_2012_q2_table']
            df_ad_2012_q3 = dict_df_ad[i]['df_adverse_ev_2012_q3_table']
            df_ad_2012_q4 = dict_df_ad[i]['df_adverse_ev_2012_q4_table']
            df_ad_clean_pre_pre = merge_ad_df(df_ad_2012_q1, df_ad_2012_q2,
                                              df_ad_2012_q3, df_ad_2012_q4) 
            dict_df_clean_ad['df_ad_clean_pre_pre'] = df_ad_clean_pre_pre
        elif i == '2013':
            df_ad_2013_q1 = dict_df_ad[i]['df_adverse_ev_2013_q1_table']
            df_ad_2013_q2 = dict_df_ad[i]['df_adverse_ev_2013_q2_table']
            df_ad_2013_q3 = dict_df_ad[i]['df_adverse_ev_2013_q3_table']
            df_ad_2013_q4 = dict_df_ad[i]['df_adverse_ev_2013_q4_table']
            df_ad_clean_pre = merge_ad_df(df_ad_2013_q1, df_ad_2013_q2,
                                              df_ad_2013_q3, df_ad_2013_q4) 
            dict_df_clean_ad['df_ad_clean_pre'] = df_ad_clean_pre
        elif i == '2014':
            df_ad_2014_q1 = dict_df_ad[i]['df_adverse_ev_2014_q1_table']
            df_ad_2014_q2 = dict_df_ad[i]['df_adverse_ev_2014_q2_table']
            df_ad_2014_q3 = dict_df_ad[i]['df_adverse_ev_2014_q3_table']
            df_ad_2014_q4 = dict_df_ad[i]['df_adverse_ev_2014_q4_table']
            df_ad_clean = merge_ad_df(df_ad_2014_q1, df_ad_2014_q2,
                                              df_ad_2014_q3, df_ad_2014_q4) 
            dict_df_clean_ad['df_ad_clean'] = df_ad_clean
    return dict_df_clean_ad


def create_ad_ev_df(year_list):
    """ load in adverse event reports from postgres sql database """
    dict_df_ad = {}
    for year in year_list:
        if year == '2012':
            dict_ad_2012 = {}
            table_list = ['df_adverse_ev_2012_q1_table',
                          'df_adverse_ev_2012_q2_table',
                          'df_adverse_ev_2012_q3_table',
                          'df_adverse_ev_2012_q4_table']
            for tab in table_list:
                dict_ad_2012[tab] = query_table_ad_ev(tab)
            dict_df_ad[year] = dict_ad_2012
        if year == '2013':
            dict_ad_2013 = {}
            table_list = ['df_adverse_ev_2013_q1_table',
                          'df_adverse_ev_2013_q2_table',
                          'df_adverse_ev_2013_q3_table',
                          'df_adverse_ev_2013_q4_table']
            for tab in table_list:
                dict_ad_2013[tab] = query_table_ad_ev(tab)
            dict_df_ad[year] = dict_ad_2013
        if year == '2014':
            dict_ad_2014 = {}
            table_list = ['df_adverse_ev_2014_q1_table',
                          'df_adverse_ev_2014_q2_table',
                          'df_adverse_ev_2014_q3_table',
                          'df_adverse_ev_2014_q4_table']
            for tab in table_list:
                dict_ad_2014[tab] = query_table_ad_ev(tab)
            dict_df_ad[year] = dict_ad_2014
    return dict_df_ad


def query_table_ad_ev(table_name):
    """ query database to get serious ad ev drug counts """
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    """ query the postgres sql database and create dataframe for serious
    adverse drug reactions"""
    sql_query = """
    SELECT drug_brand_name, drug_generic_name, drug_manuf_name,
    COUNT(serious) AS serious_count
    FROM %s
    GROUP BY drug_brand_name, drug_generic_name, drug_manuf_name;
    """
    sql_query_w_table = sql_query % (table_name)
    df = pd.read_sql_query(sql_query_w_table,con)
    return df


def merge_ad_df(df_q1, df_q2, df_q3, df_q4):
    """ merge the ad ev dataframes by drug generic name and drug brand name and
    sum the serious event count """
    df_ad_merge_q1_q2 = merge_ad_ev_tables_serious(df_q1, df_q2)
    df_ad_merge_q3_q4 = merge_ad_ev_tables_serious(df_q3, df_q4)
    df_ad_merge = merge_2_me_tables_serious(df_ad_merge_q1_q2, df_ad_merge_q3_q4)
    df_ad_clean = clean_ad_ev_table(df_ad_merge)
    df_ad_clean = df_ad_clean[['drug_generic_name_re', 'drug_brand_name_re', 'serious_count']]
    df_ad_clean.columns = ['drug_generic_name', 'drug_brand_name', 'serious_count']
    return df_ad_clean
