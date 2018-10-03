#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:04:24 2018

@author: pamelaanderson
"""
import time
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

def create_graphing_table(df_merge_ad_spending):
    table_name = 'df_spending_adverse_total'
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    ## 'engine' is a connection to a database
    ## Here, we're using postgres, but sqlalchemy can connect to other things too.
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    
    ## create a database (if it doesn't exist)
    if not database_exists(engine.url):
        create_database(engine.url)    
    df = df_merge_ad_spending
    df = df.fillna(0)
    start = time.time()
    ## insert data into database from Python (proof of concept - this won't be useful for big data, of course)
    df.to_sql(table_name, engine, chunksize=10000, if_exists='append')
    end = time.time()
    print("Time to load dataframe = ", end-start)
    engine.dispose()

def add_table_to_db(df, table_name):
    table_name = 'df_adverse_ev_2012_q4_table'
    df = df_adverse_ev
    # Define a database name
    # Set postgres username
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    ## 'engine' is a connection to a database
    ## Here, we're using postgres, but sqlalchemy can connect to other things too.
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    
    ## create a database (if it doesn't exist)
    if not database_exists(engine.url):
        create_database(engine.url)    
    
    start = time.time()
    ## insert data into database from Python (proof of concept - this won't be useful for big data, of course)
    df.to_sql(table_name, engine, chunksize=10000, if_exists='append')
    end = time.time()
    print("Time to load dataframe = ", end-start)
    engine.dispose()

def query_db_adverse_events():
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    # query:
    sql_query = """
    SELECT *
    FROM df_adverse_events_2014;
    """
    df_ad = pd.read_sql_query(sql_query,con)
    return df_ad


def query_q1_db_adverse_events():
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    # query:
    sql_query = """
    SELECT *
    FROM     df_adverse_ev_2014_q1_table;
    """
    df_ad2 = pd.read_sql_query(sql_query,con)
    return df_ad

def query_db_ad_quarter():
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    # query:
    sql_query = """
    SELECT drug_brand_name, drug_generic_name, drug_manuf_name,
    COUNT(serious) AS serious_count, 
    SUM(seriousnesscongenitalanomali) AS seriousnesscongenitalanomali_count,
    SUM(seriousnessdeath) AS seriousnessdeath_count,
    SUM(seriousnessdisabling) AS seriousnessdisabling_count,
    SUM(seriousnesshospitalization) AS seriousnesshospitalization_count,
    SUM(seriousnesslifethreatening) AS seriousnesslifethreatening_count,
    SUM(seriousnessother) AS seriousnessother_count,
    COUNT(patient_age) AS age_count
    FROM df_adverse_ev_2014_q1_table
    GROUP BY drug_brand_name, drug_generic_name, drug_manuf_name;
    """
    df_ad2 = pd.read_sql_query(sql_query,con)
    return df_ad

#    WHERE patient_age BETWEEN 65 AND 110

def query_db_ad_serious_quarter():
    dbname = 'fda_adverse_events'
    username = 'pami' # change this to your username
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    # query:
    sql_query = """
    SELECT drug_brand_name, drug_generic_name, drug_manuf_name,
    COUNT(serious) AS serious_count
    FROM df_adverse_ev_2013_q1_table
    WHERE patient_age BETWEEN 65 AND 110
    GROUP BY drug_brand_name, drug_generic_name, drug_manuf_name;
    """
    df_ad_2013_q1 = pd.read_sql_query(sql_query,con)
    return df_ad
