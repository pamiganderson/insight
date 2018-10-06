#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:31:42 2018

@author: pamelaanderson
"""

# Find adverse drug reactions for brand and generic

import time
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from functions_data_cleaning import clean_ad_ev_table
import psycopg2
import pandas as pd
import numpy as np

def query_table_patient_reaction(table_name):
    """ query the postgres sql database and create dataframe for patients
    reactions to drugs """
    sql_query = """
    SELECT drug_brand_name, drug_generic_name, drug_manuf_name, patient_react_type,
    COUNT(serious) AS serious_count
    FROM %s
    WHERE patient_age BETWEEN 65 AND 110
    GROUP BY drug_brand_name, drug_generic_name, drug_manuf_name, patient_react_type;
    """
    sql_query_w_table = sql_query % (table_name)
    df = pd.read_sql_query(sql_query_w_table,con)
    return df

# Name database and username
dbname = 'fda_adverse_events'
username = 'pami' # change this to your username

con = None
con = psycopg2.connect(database = dbname, user = username)

# Query for the adverse drug reactions for 2013
df_q1 = query_table_patient_reaction('df_adverse_ev_2013_q1_table')
df_q2 = query_table_patient_reaction('df_adverse_ev_2013_q2_table')
df_q3 = query_table_patient_reaction('df_adverse_ev_2013_q3_table')
df_q4 = query_table_patient_reaction('df_adverse_ev_2013_q4_table')

# Create pivot table for each entry on brand, generic and event
df_q1 = pd.pivot_table(df_q1, index=['drug_brand_name', 'drug_generic_name', 'drug_manuf_name', 
                                     'patient_react_type'], values = 'serious_count',
                        aggfunc = np.sum)
df_q2 = pd.pivot_table(df_q2, index=['drug_brand_name', 'drug_generic_name', 'drug_manuf_name', 
                                     'patient_react_type'], values = 'serious_count',
                        aggfunc = np.sum)
df_q3 = pd.pivot_table(df_q3, index=['drug_brand_name', 'drug_generic_name', 'drug_manuf_name', 
                                     'patient_react_type'], values = 'serious_count',
                        aggfunc = np.sum)
df_q4 = pd.pivot_table(df_q4, index=['drug_brand_name', 'drug_generic_name', 'drug_manuf_name',
                                     'patient_react_type'], values = 'serious_count',
                        aggfunc = np.sum)
# Merge all four pivot tables
df_merge = df_q1.merge(df_q2, left_index=True, right_index=True, how = 'outer')
df_merge.rename(columns={'serious_count_x': 'serious_q1', 'serious_count_y' : 'serious_q2'}, inplace=True)
df_merge = df_merge.merge(df_q3, left_index=True, right_index=True, how='outer')
df_merge.rename(columns={'serious_count': 'serious_q3'}, inplace=True)
df_merge = df_merge.merge(df_q4, left_index=True, right_index=True, how='outer')
df_merge.rename(columns={'serious_count': 'serious_q4'}, inplace=True)

# Sum the serious count 
df_merge = df_merge.fillna(0)
df_serious = df_merge[['serious_q1', 'serious_q2', 'serious_q3', 'serious_q4']]
df_serious = df_serious.sum(axis=1)

# Reset index and then allow for 
df_serious = df_serious.reset_index()

# Run through cleaning to convert important drug names
df_serious_clean = clean_ad_ev_table(df_serious)

df_serious_clean_brand = pd.pivot_table(df_serious_clean, index=['drug_brand_name',
                                                                 'drug_generic_name',
                                                                 'patient_react_type'],
                                        values = 0, aggfunc=np.sum)
df_serious_clean_brand = df_serious_clean_brand.reset_index()
