#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:43:33 2018

@author: pamelaanderson
"""

import os
import pandas as pd
import time
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


path_partd = '/Users/pamelaanderson/Documents/Insight/fda_drug_recall/medicare_partd/'

dir_list = os.listdir(path_partd)
df = pd.DataFrame()
for i in dir_list:
    print(i)
    df_i = pd.read_csv(path_partd + i)
    df = pd.concat([df, df_i], axis=0)
    
    

# Define a database name (we're using a dataset on births, so we'll call it birth_db)
# Set your postgres username
dbname = 'fda_adverse_events'
username = 'pami' # change this to your username

## 'engine' is a connection to a database
## Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print(engine.url)

start = time.time()
## insert data into database from Python (proof of concept - this won't be useful for big data, of course)
df.to_sql('df_medicare_partd_2014', engine, chunksize=10000, if_exists='append')
end = time.time()
print("Time to load one year = ", end-start)
