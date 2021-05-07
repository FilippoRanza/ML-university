#! /usr/bin/python

from argparse import ArgumentParser
import os
import pickle
import json
import datetime
import shutil

from sklearn import model_selection, neural_network, metrics, model_selection
import pandas as pd
import numpy as np


from utils import load_dataset



parser = ArgumentParser()
parser.add_argument("-f", "--feature-set", required=True)
parser.add_argument("-t", "--target-set", required=True)
args = parser.parse_args()


data = load_dataset(args.feature_set).values
target = load_dataset(args.target_set).values.ravel()

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.25)

param_grid = [
    {
        "max_iter" : [35000],
        "hidden_layer_sizes" : [
            (150, 50, 150),
            (100, 100, 100),
            (50, 100, 50),
        ],
        "learning_rate": ["adaptive", "constant", "invscaling"],
        "solver": ['lbfgs'],
        "activation": ['logistic', 'tanh', 'relu'],
        "alpha": (10.0 ** (-np.arange(1, 7)))
    }
]

grid_cls = model_selection.GridSearchCV(neural_network.MLPClassifier(), param_grid, n_jobs=-1, cv=10, verbose=3)
grid_cls.fit(x_train, y_train)
score = grid_cls.score(x_test, y_test)
print(score)
pred = grid_cls.predict(x_test)
print(metrics.classification_report(y_test, pred))
print(grid_cls.best_params_)
with open('best-nn-cls.dat', "wb") as file:
    pickle.dump(grid_cls.best_estimator_, file)
