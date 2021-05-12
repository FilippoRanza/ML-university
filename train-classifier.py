#! /usr/bin/python

from argparse import ArgumentParser
import os
import pickle
import json
import datetime
import shutil

from sklearn import tree, model_selection, ensemble, metrics, neighbors
import pandas as pd
import numpy as np


from utils import load_dataset, DiscordFrontEnd



parser = ArgumentParser()
parser.add_argument("-f", "--feature-set", required=True, nargs='+')
parser.add_argument("-t", "--target-set", required=True, nargs='+')
parser.add_argument("-o", "--output-dir", required=True)
parser.add_argument("-u", "--webhook-url")
mutex_group = parser.add_mutually_exclusive_group()
mutex_group.add_argument("--scoring", default=False, action="store_true")
mutex_group.add_argument("--loss", default=None)
args = parser.parse_args()

if len(args.feature_set) != len(args.target_set):
    print("--feature-set and --target-set require always the same number of arguments")
    exit(1)

if len(args.feature_set) > 2:
    print("--feature-set and --target-set takes 1 or 2 arguments")
    exit(1)


discord = DiscordFrontEnd(args.webhook_url)

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H-%M")
target_dir = os.path.join(args.output_dir, f"Test-{time_stamp}")

os.mkdir(target_dir)


if len(args.feature_set) == 2:
    train_feature, test_feature = args.feature_set
    train_target, test_target = args.target_set
    x_train = load_dataset(train_feature).values
    x_test = load_dataset(test_feature).values
    y_train = load_dataset(train_target).values.ravel()
    y_test = load_dataset(test_target).values.ravel()
else:
    feature_set, = args.feature_set
    target_set, = args.target_set
    data = load_dataset(feature_set).values
    target = load_dataset(target_set).values.ravel()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.25)


tree_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "splitter": ["best"],
        "min_samples_split": list(range(2, 5)),
        "max_depth": list(range(5, 15)),
        "min_samples_leaf": list(range(1, 5)),
        "max_features": [None, "sqrt", "log2"],
        "class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0:1, 1: 5}, None],
        "random_state": [42]
    }
]

forest_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "n_estimators": list(range(50, 450, 50)),
        "bootstrap": [True, False],
        "min_samples_split": list(range(2, 7)),
        "max_depth": list(range(3,  10)),
        "min_samples_leaf": list(range(1, 3)),
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
        "random_state": [42]
    }
]




extra_trees_param_grid = [
    {
        "criterion": ["gini", "entropy"],
        "n_estimators": list(range(50, 450, 50)),
        "bootstrap": [True, False],
        "min_samples_split": list(range(2, 7)),
        "max_depth": list(range(8, 18)),
        "min_samples_leaf": list(range(1, 3)),
        "max_features": [None],
        "class_weight": [None],
        "random_state": [42]
    }
]


gradient_boost_param_grid = [
    {
        "loss": ['deviance', 'exponential'],
        "learning_rate": [0.05, 0.1],
        "n_estimators": list(range(300, 550, 50)),
        "max_depth": [3, 4],
        "max_features": ["sqrt", "log2"],
        "criterion": ['friedman_mse', 'mse'],
        "min_samples_leaf": list(range(1, 5)),
        "min_samples_split": list(range(2, 5)),
        "subsample": [.5, .75, 1.0],
        "tol": [1e-2, 1e-3, 1e-4, 1e-5],
        "random_state": [42]
    }
]

ada_boost_param_grid = [
    {
        "n_estimators": list(range(50, 300, 25)),
        "learning_rate": [0.5, 1.0, 1.5],
        "algorithm": ["SAMME", "SAMME.R"],
        "random_state": [42]
    }

]


neighbors_param_grid = [
    {
        "n_neighbors" : list(range(5, 115, 10)),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "random_state": [42]
    }
]   


test_classifiers = [
    #("k-nearset-neighbors", neighbors.KNeighborsClassifier, neighbors_param_grid),
    #("ada-boost", ensemble.AdaBoostClassifier, ada_boost_param_grid),
    ("gradient-boost", ensemble.GradientBoostingClassifier, gradient_boost_param_grid),
    ("random-forest", ensemble.RandomForestClassifier, forest_param_grid),
    #("decision-tree", tree.DecisionTreeClassifier, tree_param_grid),
    ("extra-tree", ensemble.ExtraTreesClassifier, extra_trees_param_grid)
]

def scale_loss(y_true, y_pred):
    scale_y_pred = 2 * y_pred
    scale_y_true = 2 * y_true
    err = np.abs(scale_y_pred - scale_y_true)
    err = np.sum(err)
    err = int(err)
    return err

def triple_loss(y_true, y_pred):
    err = y_true - y_pred
    err = (err != 0) * 2 + err
    err = int(np.sum(err))
    return err

def precision_loss(y_true, y_pred):
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    for t, p in zip(y_true, y_pred):
        if t:
            positive += 1
        else:
            negative += 1
        
        if t != p:
            if t:
                false_negative += 1
            else:
                false_positive += 1

    positive_score = false_positive / positive
    negative_score = false_negative / negative
    return positive_score + negative_score


loss_functions = {
    'scale': scale_loss,
    'triple':triple_loss,
    'precision': precision_loss,
}

if args.scoring:
    scoring = "f1"
elif args.loss:
    try:
        loss = loss_functions[args.loss]
        scoring = metrics.make_scorer(loss, greater_is_better=False)
    except KeyError:
        print(f"unknown loss function: {args.loss}")
        print("try with", loss_functions.keys())
        exit(1)
else:
    scoring = None


for name, cls_builder, param_grid in test_classifiers:
    discord.send_message(f"Start training: {name}")
    grid_cls = model_selection.GridSearchCV(cls_builder(), param_grid, n_jobs=-1, cv=3, verbose=3, scoring=scoring)
    grid_cls.fit(x_train, y_train)

    estimator = os.path.join(target_dir, f"{name}-estimator.dat")
    with open(estimator, "wb") as file:
        pickle.dump(grid_cls.best_estimator_, file)

    y_pred = grid_cls.predict(x_test)

    report = metrics.classification_report(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    report_file = os.path.join(target_dir, f"{name}-report.txt")
    with open(report_file, "w") as file:
        print(report, file=file)
        print(conf_matrix, file=file)


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


    log_file = os.path.join(target_dir, f"{name}-log.txt")
    with open(log_file, "w") as file:
        print("Grid Search statistics:", file=file)
        means = grid_cls.cv_results_["mean_test_score"]
        stds = grid_cls.cv_results_["std_test_score"]
        values = []
        for mean, std, params in zip(means, stds, grid_cls.cv_results_["params"]):
            values.append((mean, std, params))

        values.sort(key=lambda  x: -x[0])

        for mean, std, params in values:
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=file)
        
        print(file=file)

    config_log_file = os.path.join(target_dir, f"{name}-conf.json")
    with open(config_log_file, "w") as file:
        json.dump(param_grid, file)



    discord.send_message(f"Done training: {name}")


with open(os.path.join(target_dir, 'dataset-info.txt'), "w") as file:
    print(args.feature_set, file=file)
    print(args.target_set, file=file)
    if args.scoring:
        print("using f1 score", file=file)
    if args.loss:
        print(f"using {args.loss} loss", file=file)

archive_name = os.path.join(args.output_dir, f"Test-{time_stamp}")
shutil.make_archive(archive_name, 'zip', target_dir)

archive_name += '.zip'
discord.send_file(archive_name)

shutil.rmtree(target_dir)

