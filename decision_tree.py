#! /usr/bin/python

from argparse import ArgumentParser

from sklearn import tree, impute, model_selection
import pandas as pd
import numpy as np

from utils import load_dataset

parser = ArgumentParser()
parser.add_argument('-f', '--feature-set', required=True)
parser.add_argument('-t', '--target-set', required=True)
args = parser.parse_args()

data = load_dataset(args.feature_set).values

imp = impute.SimpleImputer(missing_values=np.nan, strategy='median')
imp = imp.fit(data)

data = imp.transform(data)

target = load_dataset(args.target_set).values


tree_cls = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=3, max_depth=8, min_samples_leaf=4, ccp_alpha=0.5)


k_fold = model_selection.KFold(n_splits=10, shuffle=True)
for i, (train_index, test_index) in enumerate(k_fold.split(data, target)):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    tree_cls.fit(x_train, y_train)
    acc = tree_cls.score(x_test, y_test)
    print(f"Iter: {i} - accuracy: {acc}")

