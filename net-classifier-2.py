#! /usr/bin/python

from argparse import ArgumentParser
import pickle
import os
import shutil

import torch
from torch import nn, optim
from sklearn import model_selection, preprocessing, pipeline, metrics
from skorch import NeuralNetClassifier
import numpy as np

from utils import load_dataset, time_stamp, DiscordFrontEnd


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 150),
            nn.ReLU(),
            nn.Linear(150, 400),
            nn.ReLU(),
            nn.Linear(400, 70),
            nn.ReLU(),
            nn.Linear(70, 50),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X


class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 20),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X


class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 70),
            nn.ReLU(),
            nn.Linear(70, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X


parser = ArgumentParser()
parser.add_argument("-f", "--feature-set", required=True)
parser.add_argument("-t", "--target-set", required=True)
parser.add_argument("-u", "--url")
args = parser.parse_args()

discord = DiscordFrontEnd(args.url)

data = load_dataset(args.feature_set).values.astype(np.float32)
target = load_dataset(args.target_set).values.ravel().astype(np.int64)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.25
)


param_grid = [
    {
        "max_epochs": [1000, 1500, 2000, 2500, 3000],
        "lr": [0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007],
        "criterion": [nn.NLLLoss, nn.L1Loss, nn.CrossEntropyLoss, nn.GaussianNLLLoss],
        "optimizer": [optim.Adam, optim.SGD, optim.Adadelta],
    }
]

neural_net = [Network1, Network2, Network3]

networks = [
    (n.__name__, NeuralNetClassifier(n, device=device, iterator_train__shuffle=True))
    for n in neural_net
]



stamp = time_stamp()
dir_name = f"net-{stamp}"
os.mkdir(dir_name)
for name, net in networks:

    grid_cls = model_selection.GridSearchCV(net, param_grid, cv=4, n_jobs=-1)

    grid_cls.fit(x_train, y_train)
    score = grid_cls.score(x_test, y_test)
    y_pred = grid_cls.predict(x_test)

    res = metrics.classification_report(y_test, y_pred)
    print(score)
    print(res)


    net_file = os.path.join(dir_name, f"net-{name}.dat")
    with open(net_file, "wb") as file:
        pickle.dump(grid_cls.best_estimator_, file)

    log_file = os.path.join(dir_name, f"log-{name}.txt")
    with open(log_file, "w") as file:
        print("Best model test score:", file=file)
        print(score, file=file)
        print("Classification report:", file=file)
        print(res, file=file)
        print("Best classifier parameters:", file=file)
        print(grid_cls.best_params_, file=file)
        print(file=file)

        print("Grid Search statistics:", file=file)
        means = grid_cls.cv_results_["mean_test_score"]
        stds = grid_cls.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, grid_cls.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=file)
        print(file=file)

discord.send_message("Training Done!")
discord.send_directory(dir_name)
