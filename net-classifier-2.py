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
    def __init__(self, non_linear):
        super(Network1, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 150),
            non_linear(),
            nn.Linear(150, 400),
            non_linear(),
            nn.Linear(400, 70),
            non_linear(),
            nn.Linear(70, 50),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X

class Network11(nn.Module):
    def __init__(self, non_linear):
        super(Network1, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 150),
            non_linear(),
            nn.Linear(150, 400),
            non_linear(),
            nn.Linear(400, 70),
            non_linear(),
            nn.Linear(70, 50),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X



class Network3(nn.Module):
    def __init__(self, non_linear):
        super(Network3, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 50),
            non_linear(),
            nn.Linear(50, 40),
            non_linear(),
            nn.Linear(40, 70),
            non_linear(),
            nn.Linear(70, 50),
            non_linear(),
            nn.Linear(50, 30),
            non_linear(),
            nn.Linear(30, 30),
            nn.Softmax(dim=-1),
        )

    def forward(self, X, **kwargs):
        X = self.sequential(X)
        return X


class Network31(nn.Module):
    def __init__(self, non_linear):
        super(Network3, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(34, 50),
            non_linear(),
            nn.Linear(50, 40),
            non_linear(),
            nn.Linear(40, 70),
            non_linear(),
            nn.Linear(70, 50),
            non_linear(),
            nn.Linear(50, 30),
            non_linear(),
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
        "criterion": [nn.NLLLoss,  nn.CrossEntropyLoss],
        "optimizer": [optim.Adam, optim.SGD, optim.Adadelta],
    }
]

neural_net_class = [Network1, Network11, Network3, Network31]
non_linear = [nn.ReLU, nn.Tanh, nn.Sigmoid]

neural_net = []
for net_cls in neural_net_class:
    for nl_cls in non_linear:
        name = f"{net_cls.__name__}-{nl_cls.__name__}"
        inst = net_cls(nl_cls)
        neural_net.append(inst)

networks = [
    (n, NeuralNetClassifier(i, device=device, iterator_train__shuffle=True))
    for n, i in neural_net
]


stamp = time_stamp()
dir_name = f"net-{stamp}"
os.mkdir(dir_name)
count = len(networks)
for i, (name, net) in enumerate(networks):
    discord.send_message(f"Begin training {name} {i}/{count}")
    grid_cls = model_selection.GridSearchCV(net, param_grid, cv=4, n_jobs=-1)

    grid_cls.fit(x_train, y_train)
    score = grid_cls.score(x_test, y_test)
    y_pred = grid_cls.predict(x_test)

    res = metrics.classification_report(y_test, y_pred)
    print(score)
    print(res)

    discord.send_message(f"Results for: {name}")
    discord.send_message("Score")
    discord.send_message(score)
    discord.send_message("Result")
    discord.send_message(res)

    discord.send_message(f"Done training {name} {i}/{count}")

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
