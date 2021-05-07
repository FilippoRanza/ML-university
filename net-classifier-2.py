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

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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

discord = DiscordFrontEnd(parser.url)

data = load_dataset(args.feature_set).values.astype(np.float32)
target = load_dataset(args.target_set).values.ravel().astype(np.int64)

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.25)


net = NeuralNetClassifier(
    Network,
    max_epochs=1500,
    lr=0.01,
    device=device,
    iterator_train__shuffle=True,
)



param_grid = [
    {
        'max_epochs':[1000, 1500, 2000, 2500],
        'lr': [0.004, 0.005, 0.006],
        'criterion': [nn.NLLLoss, nn.CrossEntropyLoss, nn.MSELoss],
        'optimizer': [optim.LBFGS, optim.Adam, optim.SGD ]
    }
]

grid_cls = model_selection.GridSearchCV(net, param_grid, cv=4, n_jobs=4)

grid_cls.fit(x_train, y_train)
score = grid_cls.score(x_test, y_test)
y_pred = grid_cls.predict(x_test)

res = metrics.classification_report(y_test, y_pred)
print(score)
print(res)

stamp = time_stamp()

dir_name = f"net-{stamp}"
os.mkdir(dir_name)
net_file = os.path.join(dir_name, "net.dat")
with open(net_file, "wb") as file:
    pickle.dump(grid_cls.best_estimator_, file)

log_file = os.path.join(dir_name, "log.txt")
with open(log_file, "w") as file:
    print("Best model test score:", file=file)
    print(score, file=file)
    print("Classification report:", file=file)
    print(res, file=file)
    print("Best classifier parameters:", file=file)
    print(grid_cls.best_params_, file=file)
    print(file=file)
    
    print("Grid Search statistics:", file=file)
    means = grid_cls.cv_results_['mean_test_score']
    stds = grid_cls.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_cls.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=file)
    print(file=file)

discord.send_message("Training Done!")
discord.send_directory(dir_name)
