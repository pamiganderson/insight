#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:54:44 2018

@author: pamelaanderson
"""
import numpy as np
import pandas as pd



def find_nti_drugs(df_merge_class):
    """ Determine if drugs are narrow therapeutic index """
    nti_list = ['carbamazepine',
                'cyclosporine',
                'digoxin',
                'ethosuximide',
                'levothyroxine sodium',
                'lithium carbonate',
                'phenytoin',
                'procainamide',
                'theophylline anhydrous',
                'warfarin sodium',
                'tacrolimus']
    nti_risk = []
    for i, val in enumerate(df_merge_class.index.values):
        if val in nti_list:
            nti_risk.append(1)
        else:
            nti_risk.append(0)
    df_merge_class['nti_index'] = pd.Series(nti_risk, index=df_merge_class.index.values)
    return df_merge_class


def find_num_act_ingredients(df_merge_class):
    """ Find the number of active ingredients in drugs """
    path = '/Users/pamelaanderson/Documents/Insight/spending/'
    file_name = 'products.csv'     
    df = pd.read_csv(path+file_name)
    num_act_ingredients = []
    for i in df['ActiveIngredient']:
        num_act_ingredients.append(len(i.split(';')))
    df['num_act_ingredients'] = pd.Series(num_act_ingredients)
    df_piv = pd.pivot_table(df, index='DrugName',
                            values = 'num_act_ingredients',
                            aggfunc = np.max)
    df_piv = df_piv.reset_index()
    df_piv['DrugName'] = df_piv['DrugName'].str.lower()
    df_piv = df_piv.set_index('DrugName')
    df_merge_ingre = df_merge_class.merge(df_piv, left_index=True,
                                          right_index=True, how='left')
    num_act_ingre = df_merge_ingre['num_act_ingredients']
    num_act_ingre = num_act_ingre.fillna(1)
    df_merge_ingre['num_act_ingredients'] = num_act_ingre
    return df_merge_ingre
