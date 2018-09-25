#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
************************************************************

user_engagement_compare_models.py
Humon
Pami Anderson
05-July-2018
Copyright 2017 Humon. All rights reserved.

------------------------------------------------------------
This code uses the summary from the user_engagement_analysis
which reports a dataframe of the user_ids that have recorded 
workouts and the number of workouts they have done after the 
first week and applies various predictive models to see
if we can determine the users who will not remain engaged
with the product

Written in Python 3.6
************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
# import models and other features from sci-kit learn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def compare_classifiers(df_features, resp_var):
    # import plotting to visualize results from different models
    old_settings = np.seterr(all='ignore') 

    # set random state
    seed = 10
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    #models.append(('CART', DecisionTreeClassifier(min_samples_leaf = 0.05, max_depth = 5)))
    models.append(('CART', DecisionTreeClassifier(class_weight = 'balanced', random_state = seed, criterion = 'entropy')))
    models.append(('RF', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))

    # Prepare data
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.25,
                                                        random_state = seed)


    results = []
    names = []
    # scoring parameters: http://scikit-learn.org/stable/modules/model_evaluation.html
    scoring_param = 'roc_auc'
    for name, model in models:
        kfold = model_selection.KFold(n_splits = 5, random_state = seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = scoring_param)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # Box plot of comparison of models
    fig = plt.figure()
    fig.suptitle('Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_ylabel('ROC AUC')
    plt.show()

    return results

def random_forest_model(df_features, resp_var):
        # set random state
    seed = 10
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    #models.append(('CART', DecisionTreeClassifier(min_samples_leaf = 0.05, max_depth = 5)))
    models.append(('CART', DecisionTreeClassifier(class_weight = 'balanced', random_state = seed, criterion = 'entropy')))
    models.append(('RF', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))

    # Prepare data
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.25,
                                                        random_state = seed)


    results = []
    names = []