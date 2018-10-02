#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:18:39 2018

@author: pamelaanderson
"""

def clean_ad_ev_table(df):
    df['drug_generic_name'] = df['drug_generic_name'].str.lower()
    df['drug_brand_name'] = df['drug_brand_name'].str.lower()

    # Fix generics names
    drug_gen_series = df['drug_generic_name']
    drug_gen_series = drug_gen_series.str.replace('hydrochloride', 'hci')
    drug_gen_series = drug_gen_series.str.replace('acetaminophen', 'acetaminophen with codeine')
    drug_gen_series = drug_gen_series.str.replace('acetaminophen and codeine phosphate', 'acetaminophen with codeine')
    drug_gen_series = drug_gen_series.str.replace('acetaminophen with codeine and codeine phosphate', 'acetaminophen with codeine')
    drug_gen_series = drug_gen_series.str.replace('acetazolamide sodium', 'acetazolamide')
    drug_gen_series = drug_gen_series.str.replace('acyclovir sodium', 'acyclovir')
    drug_gen_series = drug_gen_series.str.replace('albuterol', 'albuterol sulfate')
    drug_gen_series = drug_gen_series.str.replace('alfuzosin', 'alfuzosin hci')
    drug_gen_series = drug_gen_series.str.replace('betaxolol', 'betaxolol hci')
    drug_gen_series = drug_gen_series.str.replace('carbidopa and levodopa', 'carbidopa\levodopa')
    drug_gen_series = drug_gen_series.str.replace('estradiol,', 'estradiol')
    drug_gen_series = drug_gen_series.str.replace('levetiracetam injection', 'levetiracetam')
    drug_gen_series = drug_gen_series.str.replace('lisinopril and hydrochlorothiazide', 'lisinopril/hydrochlorothiazide')
    drug_gen_series = drug_gen_series.str.replace('pantoprazole', '')
    drug_gen_series = drug_gen_series.str.replace('raloxifene', 'raloxifene hci')

    drug_gen_series[(df['drug_generic_name'] == 'metoprolol') & (df['drug_manuf_name'] != 'AstraZeneca Pharmaceuticals LP')] = 'metoprolol succinate'
    drug_gen_series[(df['drug_generic_name'] == 'montelukast') & (df['drug_manuf_name'] != 'Merck Sharp & Dohme Corp.')] = 'montelukast sodium'
    drug_gen_series[(df['drug_generic_name'] == 'pantoprazole') & (df['drug_manuf_name'] == 'Camber Pharmaceuticals, Inc.')] = 'pantoprazole sodium'
    drug_gen_series[(df['drug_generic_name'] == 'sertraline') & (df['drug_manuf_name'] != 'Pfizer Laboratories Div Pfizer Inc')] = 'sertraline hci'


    # Fix brands names
    drug_brand_series = df['drug_brand_name']
    drug_brand_series = drug_brand_series.replace('fluoxetine', 'fluoxetine hci')
    drug_brand_series = drug_brand_series.replace('hydrochloride', 'hci')
    drug_brand_series = drug_brand_series.replace('tylenol regular strength', 'Tylenol-Codeine No.3')
    drug_brand_series = drug_brand_series.replace('tylenol with codeine', 'Tylenol-Codeine No.3')
    drug_brand_series = drug_brand_series.str.replace('albuterol', 'albuterol sulfate')    
    drug_brand_series = drug_brand_series.str.replace('alfuzosin hydrochloride', 'alfuzosin hci er')
    drug_brand_series = drug_brand_series.str.replace('aloprim', 'allopurinol')
    drug_brand_series = drug_brand_series.str.replace('amlodipine besylate 10 mg', 'amlodipine besylate')
    drug_brand_series[(df['drug_generic_name'] == 'atenolol') & (df['drug_manuf_name'] != 'AstraZeneca Pharmaceuticals LP')] = 'atenolol'
    drug_brand_series = drug_brand_series.str.replace('betaxolol', 'betaxolol hci')
    drug_brand_series[(df['drug_generic_name'] == 'casodex') & (df['drug_manuf_name'] != 'AstraZeneca Pharmaceuticals LP')] = 'bicalutamide'
    drug_brand_series = drug_brand_series.str.replace('brimonidine', 'brimonidine tartrate')
    drug_brand_series[(df['drug_generic_name'] == 'candesartan cilexetil') & (df['drug_manuf_name'] != 'AstraZeneca Pharmaceuticals LP')] = 'candesartan cilexetil'
    drug_brand_series[(df['drug_generic_name'] == 'carbamazepine') & (df['drug_manuf_name'] != 'Novartis Pharmaceuticals Corporation')] = 'carbamazepine'
    drug_brand_series = drug_brand_series.str.replace('carisoprodol immediate release', 'carisoprodol')
    drug_brand_series = drug_brand_series.str.replace('celecoxib 50 mg', 'celecoxib')
    drug_brand_series = drug_brand_series.str.replace('cipro hc', 'cipro')
    drug_brand_series[(df['drug_generic_name'] == 'clopidogrel bisulfate') & (df['drug_manuf_name'] != 'Bristol-Myers Squibb/Sanofi Pharmaceuticals Partnership')] = 'clopidogrel bisulfate'
    drug_brand_series = drug_brand_series.str.replace('diltiazem hci extended release', 'diltiazem hci')
    drug_brand_series = drug_brand_series.str.replace('diltiazem hci extended release', 'diltiazem hci')
    drug_brand_series = drug_brand_series.str.replace('duloxetine delayed-release', 'duloxetine hci')
    drug_brand_series[(df['drug_generic_name'] == 'enoxaparin sodium') & (df['drug_manuf_name'] != 'sanofi-aventis U.S. LLC')] = 'enoxaparin sodium'
    drug_brand_series[(df['drug_generic_name'] == 'exemestane') & (df['drug_manuf_name'] != 'Pharmacia and Upjohn Company LLC')] = 'exemestane'
    drug_brand_series = drug_brand_series.str.replace('fentanyl - novaplus', 'fentanyl')
    drug_brand_series[(df['drug_generic_name'] == 'furosemide') & (df['drug_manuf_name'] != 'Sanofi-Aventis U.S. LLC')] = 'furosemide'
    drug_brand_series = drug_brand_series.str.replace('gabapentin kit', 'gabapentin')
    drug_brand_series = drug_brand_series.str.replace('gentamicin', 'gentamicin sulfate')
    drug_brand_series[(df['drug_generic_name'] == 'glimepiride') & (df['drug_manuf_name'] != 'Sanofi-Aventis U.S. LLC')] = 'glimepiride'
    drug_brand_series[(df['drug_generic_name'] == 'irbesartan') & (df['drug_manuf_name'] != 'sanofi-aventis U.S. LLC')] = 'irbesartan'
    drug_brand_series[(df['drug_generic_name'] == 'lamotrigine') & (df['drug_manuf_name'] != 'GlaxoSmithKline LLC')] = 'lamotrigine'
    drug_brand_series = drug_brand_series.str.replace('health mart lansoprazole', 'lansoprazole')
    drug_brand_series= drug_brand_series.str.replace('levetiracetam levetiracetam', 'levetiracetam')
    drug_brand_series[(df['drug_generic_name'] == 'levofloxacin') & (df['drug_manuf_name'] != 'Janssen Pharmaceuticals, Inc.')] = 'levofloxacin'
    drug_brand_series[(df['drug_generic_name'] == 'lisinopril') & (df['drug_manuf_name'] != 'Merck Sharp & Dohme Corp.')] = 'lisinopril'
    drug_brand_series = drug_brand_series.str.replace('acetaminophen and codeine', 'Acetaminophen-Codeine')
    drug_brand_series[(df['drug_generic_name'] == 'acyclovir sodium') & (df['drug_manuf_name'] != 'Prestium Pharma, Inc.')] = 'acyclovir'
    drug_brand_series[(df['drug_generic_name'] == 'lisinopril') & (df['drug_manuf_name'] != 'LUPIN LIMITED')] = 'lisinopril'
    drug_brand_series[(df['drug_generic_name'] == 'lisinopril/hydrochlorothiazide') & (df['drug_manuf_name'] != 'Almatica Pharma Inc.')] = 'lisinopril/hydrochlorothiazide'
    drug_brand_series[(df['drug_generic_name'] == 'azacitidine') & (df['drug_manuf_name'] != 'Celgene Corporation')] = 'azacitidine'


    drug_brand_series = drug_brand_series.str.replace('methotrexate', 'methotrexate sodium')
    drug_brand_series = drug_brand_series.str.replace('montelukast sodium chewable', 'montelukast sodium')
    drug_brand_series = drug_brand_series.str.replace('nevirapine extended release', 'nevirapine')
    drug_brand_series = drug_brand_series.str.replace('raloxifene', 'raloxifene hci')
    drug_brand_series = drug_brand_series.str.replace('being well heartburn relief', 'ranitidine hci')
    drug_brand_series = drug_brand_series.str.replace('acid reducer', 'ranitidine hci')

    drug_brand_series[(df['drug_generic_name'] == 'meloxicam') & (df['drug_manuf_name'] != 'Boehringer Ingelheim Pharmaceuticals Inc.')] = 'meloxicam'
    drug_brand_series[(df['drug_generic_name'] == 'memantine') & (df['drug_manuf_name'] != 'Allergan, Inc.')] = 'memantine'
    drug_brand_series[(df['drug_generic_name'] == 'metaxalone') & (df['drug_manuf_name'] != 'Pfizer Laboratories Div Pfizer Inc')] = 'metaxalone'
    drug_brand_series[(df['drug_generic_name'] == 'metoprolol') & (df['drug_manuf_name'] != 'AstraZeneca Pharmaceuticals LP')] = 'metoprolol succinate'
    drug_brand_series[(df['drug_generic_name'] == 'modafinil') & (df['drug_manuf_name'] != 'Cephalon, Inc.')] = 'modafinil'
    drug_brand_series[(df['drug_generic_name'] == 'nateglinide') & (df['drug_manuf_name'] != 'Novartis Pharmaceuticals Corporation')] = 'nateglinide'
    drug_brand_series[(df['drug_generic_name'] == 'nitroglycerin') & (df['drug_manuf_name'] != 'Pfizer Laboratories Div Pfizer Inc')] = 'nitroglycerin'
    drug_brand_series[(df['drug_generic_name'] == 'oxcarbazepine') & (df['drug_manuf_name'] != 'Novartis Pharmaceuticals Corporation')] = 'oxcarbazepine'
    drug_brand_series[(df['drug_generic_name'] == 'oxycodone hci') & (df['drug_manuf_name'] != 'Mallinckrodt ARD Inc.')] = 'oxycodone hci'
    drug_brand_series[(df['drug_generic_name'] == 'paclitaxel') & (df['drug_manuf_name'] != 'Celgene Corporation')] = 'paclitaxel'
    drug_brand_series[(df['drug_generic_name'] == 'pantoprazole sodium') & (df['drug_manuf_name'] != 'Wyeth Pharmaceuticals LLC, a subsidiary of Pfizer Inc.')] = 'pantoprazole sodium'
    drug_brand_series[(df['drug_generic_name'] == 'phenytoin') & (df['drug_manuf_name'] != 'Pfizer Laboratories Div Pfizer Inc')] = 'phenytoin'
    drug_brand_series[(df['drug_generic_name'] == 'ranitidine hci') & (df['drug_manuf_name'] != 'GlaxoSmithKline Consumer Healthcare Holdings (US) LLC')] = 'ranitidine hci'
    drug_brand_series[(df['drug_generic_name'] == 'sertraline') & (df['drug_manuf_name'] != 'Pfizer Laboratories Div Pfizer Inc')] = 'sertraline hci'

    df['drug_generic_name_re'] = drug_gen_series
    df['drug_brand_name_re'] = drug_brand_series
    return df