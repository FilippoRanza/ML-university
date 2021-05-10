#! /usr/bin/python

from argparse import ArgumentParser
import os
import pickle
import json
import datetime
import shutil

from sklearn import tree, model_selection, ensemble, metrics
import pandas as pd
import numpy as np


from utils import load_dataset, DiscordFrontEnd



parser = ArgumentParser()
parser.add_argument("-f", "--feature-set", required=True)
parser.add_argument("-t", "--target-set", required=True)
parser.add_argument("-o", "--output-dir", required=True)
parser.add_argument("-u", "--webhook-url")
args = parser.parse_args()

discord = DiscordFrontEnd(args.webhook_url)

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H-%M")
target_dir = os.path.join(args.output_dir, f"Test-{time_stamp}")

os.mkdir(target_dir)

data = load_dataset(args.feature_set).values


target = load_dataset(args.target_set).values.ravel()

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.25)


tree_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "splitter": ["best"],
        "min_samples_split": list(range(2, 5)),
        "max_depth": list(range(3, 10)),
        "min_samples_leaf": list(range(1, 5)),
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha" : np.linspace(0, 0.1, 10),
        "min_weight_fraction_leaf": np.linspace(0, 0.5, 5),
    }
]

forest_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "n_estimators": list(range(100, 1100, 100)),
        "bootstrap": [True, False],
        "min_samples_split": list(range(2, 5)),
        "max_depth": list(range(3, 8)),
        "min_samples_leaf": list(range(1, 5)),
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha" : np.linspace(0, 0.1, 5),
        "min_weight_fraction_leaf": np.linspace(0, 0.5, 5),
    }
]




extra_trees_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "n_estimators": list(range(100, 1100, 100)),
        "bootstrap": [True, False],
        "min_samples_split": list(range(2, 5)),
        "max_depth": list(range(3, 8)),
        "min_samples_leaf": list(range(1, 5)),
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha" : np.linspace(0, 0.1, 5),
        "min_weight_fraction_leaf": np.linspace(0, 0.5, 5),
    }
]


test_classifiers = [
    ("random-forest", ensemble.RandomForestClassifier, forest_param_grid),
    ("decision-tree", tree.DecisionTreeClassifier, tree_param_grid),
    ("extra-tree", ensemble.ExtraTreesClassifier, extra_trees_param_grid)
]

for name, cls_builder, param_grid in test_classifiers:
    discord.send_message(f"Start training: {name}")
    grid_cls = model_selection.GridSearchCV(cls_builder(), param_grid, n_jobs=-1, cv=10, verbose=3)
    grid_cls.fit(x_train, y_train)

    estimator = os.path.join(target_dir, f"{name}-estimator.dat")
    with open(estimator, "wb") as file:
        pickle.dump(grid_cls.best_estimator_, file)

    y_pred = grid_cls.predict(x_test)

    report = metrics.classification_report(y_test, y_pred)
    report_file = os.path.join(target_dir, f"{name}-report.txt")
    with open(report_file, "w") as file:
        print(report, file=file)


    log_file = os.path.join(target_dir, f"{name}-result.json")
    test_score = grid_cls.score(x_test, y_test)
    test_results = {
        'method': cls_builder.__name__,
        'params': grid_cls.best_params_,
        'train-score': grid_cls.best_score_,
        'test-score': test_score
    }
    with open(log_file, "w") as file:
        json.dump(test_results, file)

    discord.send_message(f"Done training: {name}")


with open(os.path.join(target_dir, 'dataset-info.txt'), "w") as file:
    print(args.feature_set, file=file)
    print(args.target_set, file=file)


archive_name = os.path.join(args.output_dir, f"Test-{time_stamp}")
shutil.make_archive(archive_name, 'zip', target_dir)

archive_name += '.zip'
discord.send_file(archive_name)

shutil.rmtree(target_dir)

