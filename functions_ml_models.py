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
import pandas as pd
import matplotlib.pyplot as plt
# import models and other features from sci-kit learn
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from functions_plotting import plot_feature_importance
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE, ADASYN


def compare_classifiers(df_features, resp_var):
    # import plotting to visualize results from different models
    old_settings = np.seterr(all='ignore') 

    # set random state
    seed = 42
    # prepare models
    models = []
#    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(class_weight = 'balanced', max_depth = 5)))
    #models.append(('CART', DecisionTreeClassifier(class_weight = 'balanced', criterion = 'entropy')))
    models.append(('RF', RandomForestClassifier(class_weight = 'balanced', criterion = 'entropy', max_depth = 5)))
    models.append(('NB', GaussianNB()))

    # Prepare data
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify=y,
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
    
    roc_dict = {}
    roc_list = []
    recall_list = []
    precision_list = []
    # Prepare data
    for i in range(0,25):
        seed = i
        X = df_features
        y = resp_var
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.3,
                                                            stratify=y,
                                                            random_state = seed)
        
        X_train_res, y_train_res = SMOTE(kind='borderline1').fit_sample(X_train, y_train)

        # Random Forest 
        rf = RandomForestClassifier(class_weight = 'balanced', criterion = 'entropy')
        # to see hyperparameters names: rf.get_params()
        params_rf = {'min_samples_leaf' : [0.02, 0.05, 0.1, 0.15, 0.2],
                     'max_depth' : [4, 5, 6, 7, 8]}
        
        grid_rf = GridSearchCV(estimator = rf, 
                               param_grid = params_rf, 
                               cv=5, scoring = 'recall_weighted')
        #grid_rf.fit(X_train, y_train)
        grid_rf.fit(X_train_res, y_train_res)
        rf_best_model = grid_rf.best_estimator_
        y_pred = rf_best_model.predict(X_test)
        class_report = classification_report(y_test, y_pred)
        confusion_matrix(y_test, y_pred)
        
        # Find best model parameters
#        rf_best_model = grid_rf.best_estimator_
#        y_pred = rf_best_model.predict(X_test)
#        class_report = classification_report(y_test, y_pred)
#        confusion_matrix(y_test, y_pred)
#    
        #plot important features
        importances_rf = pd.Series(rf_best_model.feature_importances_,
                                   index = X.columns)
        sorted_importances_rf = importances_rf.sort_values()
        graph_title = 'Random Forest Important Features'
        plot_feature_importance(sorted_importances_rf, graph_title)
        
        dict_results= {'confusion_matrix' : confusion_matrix(y_test, y_pred),
                       'classification_report' : classification_report(y_test, y_pred),
                       'best_model_parameters' : grid_rf.best_params_,
                       'best_model' : grid_rf.best_estimator_,
                       'best_roc_score' : grid_rf.best_score_
                       }
        roc_score = roc_auc_score(y_test, y_pred)
        roc_dict[i] = dict_results
        roc_list.append(roc_score)
        recall_list.append(recall_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        
    return dict_results