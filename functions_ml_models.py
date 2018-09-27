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

def random_forest_regressor(df_features, resp_var):

    import seaborn as sns
    from scipy import stats
    from scipy.stats import norm
    sns.distplot(resp_var,fit=norm);
    fig = plt.figure()
    res = stats.probplot(resp_var, plot=plt)
    
    # Prepare data
    seed = 42
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.25,
                                                        random_state = seed)
    
    # DECISION TREE Feature importance
    dt = rfr()
    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i in estimators:
        model = rfr(n_estimators=i,max_depth=None)
        scores_rfr = cross_val_score(model,X,y,cv=10,scoring='explained_variance')
        print('estimators:',i)

    # to see hyperparameters names: dt.get_params()
    params_dt = {'min_samples_leaf' : [0.05, 0.1, 0.15],
                 'max_depth' : [3, 4, 5, 6, 7]}
    
    grid_dt = GridSearchCV(estimator = dt, 
                           param_grid = params_dt, 
                           cv=5, scoring = 'roc_auc')
    grid_dt.fit(X_train, y_train)
    
    # Find best model parameters
    dt_best_model = grid_dt.best_estimator_
    y_pred = dt_best_model.predict(X_test)
    classification_report(y_test, y_pred)
    confusion_matrix(y_test, y_pred)

    #plot important features
    importances_dt = pd.Series(dt_best_model.feature_importances_,
                               index = X.columns)
    sorted_importances_dt = importances_dt.sort_values()
    graph_title = 'Decision Tree Important Features'
    plot_feature_importance(sorted_importances_dt, graph_title)
    
    dict_results= {'confusion_matrix' : confusion_matrix(y_test, y_pred),
                   'classification_report' : classification_report(y_test, y_pred),
                   'best_model_parameters' : grid_dt.best_params_,
                   'best_model' : grid_dt.best_estimator_,
                   'best_roc_score' : grid_dt.best_score_
                   }
    return dict_results

def random_forest_model(df_features, resp_var):

    # Prepare data
    seed = 42
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.25,
                                                        stratify=y,
                                                        random_state = seed)
    
    # DECISION TREE Feature importance
    dt = RandomForestClassifier(class_weight = 'balanced', random_state = seed, criterion = 'entropy')
    # to see hyperparameters names: dt.get_params()
    params_dt = {'min_samples_leaf' : [0.05, 0.1, 0.15],
                 'max_depth' : [3, 4, 5, 6, 7]}
    
    grid_dt = GridSearchCV(estimator = dt, 
                           param_grid = params_dt, 
                           cv=5, scoring = 'recall')
    grid_dt.fit(X_train, y_train)
    
    # Find best model parameters
    dt_best_model = grid_dt.best_estimator_
    y_pred = dt_best_model.predict(X_test)
    classification_report(y_test, y_pred)
    confusion_matrix(y_test, y_pred)

    #plot important features
    importances_dt = pd.Series(dt_best_model.feature_importances_,
                               index = X.columns)
    sorted_importances_dt = importances_dt.sort_values()
    graph_title = 'Decision Tree Important Features'
    plot_feature_importance(sorted_importances_dt, graph_title)
    
    dict_results= {'confusion_matrix' : confusion_matrix(y_test, y_pred),
                   'classification_report' : classification_report(y_test, y_pred),
                   'best_model_parameters' : grid_dt.best_params_,
                   'best_model' : grid_dt.best_estimator_,
                   'best_roc_score' : grid_dt.best_score_
                   }
    return dict_results