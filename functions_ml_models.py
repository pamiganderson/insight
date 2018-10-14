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
from xgboost import XGBClassifier
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
    """ Compare different classification models --> no hyperparameter optimization"""
    # import plotting to visualize results from different models
    old_settings = np.seterr(all='ignore') 

    # set random state
    seed = 2
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
#    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('XGB', XGBClassifier()))

    # Prepare data
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify=y,
                                                        test_size = 0.3,
                                                        random_state = seed)

    sm = SMOTE(random_state=12, ratio = 1.0)
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    results = []
    names = []
    # scoring parameters: http://scikit-learn.org/stable/modules/model_evaluation.html
    scoring_param = 'average_precision'
    for name, model in models:
        kfold = model_selection.KFold(n_splits = 5, random_state = seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, x_train_res, y_train_res, cv=kfold,
                                                     scoring = scoring_param)
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
    ax.set_ylabel('P-R AUC')
    plt.show()
    

def random_forest_model(df_features, resp_var):
    """ Utilize a random forest model """    
    roc_dict = {}
    roc_list = []
    recall_list = []
    precision_list = []
    # Prepare data
    for i in range(0,25):
        seed = 2
        X = df_features
        y = resp_var
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.3,
                                                            stratify=y,
                                                            random_state = seed)
        
        # SMOTE class balancing
        sm = SMOTE(random_state=12, ratio = 1.0)
        x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
        
        # Random Forest 
        rf = RandomForestClassifier(criterion = 'entropy')
        # Number of trees
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Minimum samples to split a node
        min_samples_split = [2, 5, 10]
        # Minimum samples per leaf
        min_samples_leaf = [1, 2, 4]
        # Other parameters: bootstrap - method to determine samples in each tree
        # max_features - # featuers to consider at each split
        
        # to see hyperparameters names: rf.get_params()
        params_rf = {
                'n_estimators' : n_estimators,
                'max_features' : ['auto', 'sqrt'],
                'min_samples_leaf' : min_samples_leaf,
                'min_samples_split' : min_samples_split,
                'bootstrap' : [True, False]}
        
        grid_rf = GridSearchCV(estimator = rf, 
                               param_grid = params_rf, 
                               cv=5, scoring = 'average_precision')
        #grid_rf.fit(X_train, y_train)
        grid_rf.fit(x_train_res, y_train_res)
        rf_best_model = grid_rf.best_estimator_
        y_pred = rf_best_model.predict(X_test)
        class_report = classification_report(y_test, y_pred)
        confusion_matrix(y_test, y_pred)
            
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


def xgboost_model():
    """ Utilize a XGBoost model """
    # To prevent XG boost error    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    dict_conf = {}

    seed = 18
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3,
                                                        stratify=y,
                                                        random_state = seed)

    param_test1 = {
    'max_depth':range(3,7,2),
    'min_child_weight':range(1,2,2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                      n_estimators=1000, 
                                                      max_depth=5,
                                                      min_child_weight=1,
                                                      gamma=0, 
                                                      subsample=0.8, 
                                                      colsample_bytree=0.8,
                                                      objective= 'binary:logistic', nthread=4,
                                                      scale_pos_weight=1, seed=i), 
                            param_grid = param_test1, 
                            scoring='recall_weighted',
                            n_jobs=4,
                            iid=False, 
                            cv=5)
    
    gsearch1.fit(X_train,y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    
    xgb_best_model = gsearch1.best_estimator_
    y_pred = xgb_best_model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    confusion_matrix(y_test, y_pred)

    
def plot_roc_curve(classifier, X, y):
    """ Function to plot an ROC curve knowing the features, resp and model """
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp
    
    cv = StratifiedKFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    
    X = df_features.reset_index(drop=True)
    y = resp_var.reset_index(drop=True)
    
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_precision_recall(classifier, X_test, y_test, y_score):
    """ Function to plot an precision-recall curve knowing the features, resp and model """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature
    from sklearn.metrics import average_precision_score
    

    
    classifier.fit(X_train, y_train)
    predictions_probs = classifier.predict_proba(X_test)
    average_precision = average_precision_score(y_test, predictions_probs[:,1])

    print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
    
    precision, recall, _ = precision_recall_curve(y_test, predictions_probs[:,1])
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
    
def rf_regressor(X, y):
    """ Function to perform a random forest regressor model on data """"
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(random_state = 42)
    
    seed = i
    X = df_features
    y = resp_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3,
                                                        random_state = seed)

    # Train the model on training data
    rf.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test['chi_sq_val'])
    
    plt.figure()
    plt.plot(y_test, predictions, 'b*')
    plt.xlabel('Test Set Chi Sq')
    plt.ylabel('Predicted Chi Sq')
    plt.title('Random Forest Regressor on Chi Sq Stat')

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
