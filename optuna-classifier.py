"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

import optuna
from optuna.trial import TrialState

import numpy as np

from utils import load_dataset, time_stamp

device = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device(device)
BATCHSIZE = 64
CLASSES = 2
DIR = os.getcwd()
MIN_EPOCHS = 10
MAX_EPOCHS = 100
LOG_INTERVAL = 10

FEATURES = 33

DB_FILE = 'sqlite:///optuna_study.db'

TRAIN_FEATURE = "train-covid-core-impute-no-age.csv"
TRAIN_TARGET = "train-covid-target-set.csv"

TEST_FEATURE = "test-covid-core-impute-no-age.csv"
TEST_TARGET = "test-covid-target-set.csv"



def load_datasets(features, target):
    x = load_dataset(features).values.astype(np.float32)
    y = load_dataset(target).values.ravel().astype(np.int64)
    return x, y

class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.x, self.y = load_datasets(features, target)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y


def get_dataset():
    train_set = CustomDataset(TRAIN_FEATURE,TRAIN_TARGET) 
    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
    test_set = CustomDataset(TEST_FEATURE, TEST_TARGET)
    test_loader = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True)
    return train_loader, test_loader


def get_score(model):
    cpu_dev = torch.device("cpu")
    model.to(cpu_dev)
    model.eval()
    x, y_true = load_datasets(TEST_FEATURE, TEST_TARGET)
    x = torch.from_numpy(x)
    with torch.no_grad():
        y_pred = model(x)
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(axis=1)

    repo = metrics.classification_report(y_true, y_pred)
    return repo



def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 12)
    layers = []

    in_features = FEATURES
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.75)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

class Objective:
    def __init__(self):
        self.models = {}

    def get_best(self, trial_number):

        return self.models.get(trial_number)

    def objective(self, trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adadelta"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        train_loader, valid_loader = get_dataset()


        epochs = trial.suggest_int("n_epochs", MIN_EPOCHS, MAX_EPOCHS)

        # Training of the model.
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):

                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    # Get the index of the max log-probability.
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / len(valid_loader.dataset)

            trial.report(accuracy, epoch)
            self.models[trial.number] = model

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy





if __name__ == "__main__":

   
    study = optuna.create_study(direction="maximize", study_name="classify infection", storage=DB_FILE, load_if_exists=True)
    
    obj = Objective()
    study.optimize(obj.objective, n_trials=7500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    log_file_name = f"net-log_{time_stamp()}.txt"
    with open(log_file_name, "w") as file:
        print("Study statistics: ", file=file)
        print("  Number of finished trials: ", len(study.trials), file=file)
        print("  Number of pruned trials: ", len(pruned_trials), file=file)
        print("  Number of complete trials: ", len(complete_trials), file=file)

        print("Best trial:", file=file)
        trial = study.best_trial

        print("  Value: ", trial.value, file=file)
        print("  Number: ", trial.number, file=file)
        print("  Params: ", file=file)
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value), file=file)

        if model := obj.get_best(trial.number):
            repo = get_score(model)
            print(repo, file=file)
            file_name = f"net-model_{time_stamp()}.dat"
            with open(file_name, "wb") as file:
                torch.save(model, file)
        else:
            print("Best is from a previous run", file=file)