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
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor



DATAPATH = ""

def load_data(data_path=DATAPATH):
    
    fnc = pd.read_csv(os.path.join(data_path, "fnc.csv"))
    loading = pd.read_csv(os.path.join(data_path, "loading.csv"))
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

# Prediction time!

train, test, features, targets = load_data()

weights = [.3, .175, .175, .175, .175]

linear_models = []

linear_models.append(("Ridge regression",Ridge()))
linear_models.append(("Lasso regression", Lasso()))
linear_models.append(("Elastic net", ElasticNet()))
linear_models.append(("Least angle regression", Lars()))

kernel_models = []

kernel_models.append(("Linear Support Vector Regression", LinearSVR()))
kernel_models.append(("Support Vector Regression", SVR()))
kernel_models.append(("Nu Support Vector Regression", NuSVR()))

neighborhood_models = [("K-nearest neighbours Ball", KNeighborsRegressor(algorithm='ball_tree'))]

gaussian_models = [("Gaussian Process Regression", GaussianProcessRegressor())]

ensemble_models = []

ensemble_models.append(("Random forest", RandomForestRegressor()))
ensemble_models.append(("AdaBoost", AdaBoostRegressor()))
ensemble_models.append(("GradientBoosting", GradientBoostingRegressor()))




model_families = [("Linear Models",linear_models), ("Ensemble methods",ensemble_models),
                  ("Kernel Methods", kernel_models), ("Gaussian Methods",gaussian_models),
                  ("Neighbourhood", neighborhood_models)]

for family_name,models in model_families:
    
    print('*****{}*****'.format(family_name))

    for name, model in models:
        
        print("---{}---".format(name))
        print(model)
        
        overall_score=0
        
        for target, weight in zip(targets, weights):
            print(target,' ',weight)
            non_zero_train=train[train[target].notnull()]
            
            target_pred = cross_val_predict(model, non_zero_train[features], non_zero_train[target],
                                            cv=5, n_jobs=-1)
            
            score = metric(non_zero_train[target], target_pred)
            
            overall_score += score*weight
            
            print("{}: {}".format(target,score))
        print()
    print()
        

    
