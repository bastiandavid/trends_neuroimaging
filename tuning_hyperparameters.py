#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:54:04 2020

@author: bdavid
"""

# general imports
import os
import pandas as pd
import numpy as np

# ML imports
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor



DATAPATH = ""
FNC_SCALE = 1/500


def load_data(data_path=DATAPATH):
    
    fnc = pd.read_csv(os.path.join(data_path, "fnc.csv"))
    loading = pd.read_csv(os.path.join(data_path, "loading.csv"))
    
    fnc_features = list(fnc.columns[1:])
    fnc[fnc_features] *= FNC_SCALE
    
    features_df = fnc.merge(loading, on='Id')
    train_scores = pd.read_csv(os.path.join(data_path, "train_scores.csv"))
    train_scores["train_set"] = True
    train = features_df.merge(train_scores, on="Id", how="left")
    
    test = train[train["train_set"] != True].copy().drop(columns="train_set")
    train = train[train["train_set"] == True].copy().drop(columns="train_set")
    
    features = list(fnc.columns[1:]) + list(loading.columns[1:])
    targets = list(train_scores.columns[1:-1:])
    
    return train, test, features, targets

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

norm_abs_error = make_scorer(metric, greater_is_better=False)

# Prediction time!

train, test, features, targets = load_data()

weights = [.3, .175, .175, .175, .175]

model = GradientBoostingRegressor()

distributions = dict(loss=['ls','lad'], learning_rate=[0.05, 0.1, 0.15],
                     n_estimators=[100, 150, 200], max_depth=[3,4,5])

for target in targets:

    non_zero_train=train[train[target].notnull()]
    
    gs_clf = GridSearchCV(model, distributions, n_jobs=-1, 
                                     scoring=norm_abs_error, refit=True, cv=5,
                                     verbose=2)
    
    gs = gs_clf.fit(non_zero_train[features], y=non_zero_train[target])
    
    
    print("Finished {} grid search".format(target))
    print("Best extimator: {}".format(gs.best_estimator_))
    print("Best parameters: {}".format(gs.best_parameters_))
    print("Best score: {]".format(gs.best_score_))
    
    joblib.dump(gs, 'gs_GradientBoosting_'+target+'.pkl')
        

    
