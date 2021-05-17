from argparse import ArgumentParser
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

import optuna
from optuna.trial import TrialState

import numpy as np

import yaml

from utils import load_dataset, time_stamp, DiscordFrontEnd

device = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device(device)
DIR = os.getcwd()


def get_mandatory_config_param(conf: dict, key: str):
    try:
        output = conf[key]
    except KeyError:
        raise ValueError(f"Missing configuration parameter {key}")
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    with open(args.config_file) as file:
        data = yaml.safe_load(file)

    BATCHSIZE = data.get("BATCHSIZE", 64)
    CLASSES = get_mandatory_config_param(data, "CLASSES")
    MIN_EPOCHS = data.get("MIN_EPOCHS", 10)
    MAX_EPOCHS = data.get("MAX_EPOCHS", 100)
    LOG_INTERVAL = data.get("LOG_INTERVAL", 10)
    FEATURES = get_mandatory_config_param(data, "FEATURES")
    DB_FILE = data.get("DB_FILE", None)
    TRAIN_FEATURE = get_mandatory_config_param(data, "TRAIN_FEATURE")
    TRAIN_TARGET = get_mandatory_config_param(data, "TRAIN_TARGET")
    TEST_FEATURE = get_mandatory_config_param(data, "TEST_FEATURE")
    TEST_TARGET = get_mandatory_config_param(data, "TEST_TARGET")
    N_TRIALS = data.get("N_TRIALS", 10)
    DISCORD_URL = data.get("DISCORD_URL", None)
    STUDY_NAME = get_mandatory_config_param(data, "STUDY_NAME")
    discord = DiscordFrontEnd(DISCORD_URL)


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
    train_set = CustomDataset(TRAIN_FEATURE, TRAIN_TARGET)
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
    n_layers = trial.suggest_int("n_layers", 1, 25)
    layers = []

    in_features = FEATURES
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 192)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.75)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def suggest_weights(trial, count):
    weights = [trial.suggest_float(f"weight_{i}", 0.5, 100) for i in range(count)]
    weights = np.array(weights, dtype=np.float32)
    return weights


class ComputeProportionaWeight:
    def __init__(self):
        self.targets = {}

    def add_loader(self, loader):
        for _, targets in loader:
            for t in targets.numpy():
                self.add_value(t)
    
    def add_value(self, v):
        try:
            self.targets[v] += 1
        except KeyError:
            self.targets[v] = 1
    
    def get_weights(self):
        total = 0
        for v in self.targets.values():
            total += v
        output = np.zeros(len(self.targets), dtype=np.float32)
        for k, v in self.targets.items():
            output[k] = 1 - (v / total)
       
        return output


 
class Objective:
    def __init__(self):
        self.models = {}

    def get_best(self):
        trial = 0
        accuracy = 0
        model = None
        for k, (m, a) in self.models.items():
            if a > accuracy:
                accuracy = a
                trial = k
                model = m
        return (trial, model)

    def objective(self, trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.

        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD", "Adadelta", "Rprop"]
        )
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        train_loader, valid_loader = get_dataset()
        compute_propotional_weight = ComputeProportionaWeight()
        compute_propotional_weight.add_loader(train_loader)
        compute_propotional_weight.add_loader(valid_loader)
        proportional_weights = compute_propotional_weight.get_weights()
        proportional_weights = torch.from_numpy(proportional_weights)
        proportional_weights = proportional_weights.to(DEVICE)

        epochs = trial.suggest_int("n_epochs", MIN_EPOCHS, MAX_EPOCHS)

   



        loss_function_name = trial.suggest_categorical("loss function", ["nll_loss", "cross_entropy"])
        loss_function = getattr(F, loss_function_name)


        # Training of the model.
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target, weight=proportional_weights)
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
            self.models[trial.number] = (model, accuracy)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy


if __name__ == "__main__":
    discord.send_message("Start OPTUNA study")
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=DB_FILE,
        load_if_exists=True,
    )

    obj = Objective()
    study.optimize(obj.objective, n_trials=N_TRIALS)
    discord.send_message(f"Done {N_TRIALS} trials")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    dir_name = f"optuna-optim-{time_stamp()}"
    os.mkdir(dir_name)

    log_file_name = f"net-log_{time_stamp()}.txt"
    log_file_path = os.path.join(dir_name, log_file_name)
    with open(log_file_path, "w") as file:
        print("Study statistics: ", file=file)
        print("  Number of finished trials: ", len(study.trials), file=file)
        print("  Number of pruned trials: ", len(pruned_trials), file=file)
        print("  Number of complete trials: ", len(complete_trials), file=file)

        print("Best trial:", file=file)
        trial = study.best_trial

        print("  Value: ", trial.value, file=file)
        print("  Global Best Number: ", trial.number, file=file)
        print("  Params: ", file=file)
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value), file=file)

        trial, model = obj.get_best()
        repo = get_score(model)
        print(f"Local Best Number: {trial}", file=file)
        print(repo, file=file)
        file_name = f"net-model_{time_stamp()}.dat"
        model_file_path = os.path.join(dir_name, file_name)
        with open(model_file_path, "wb") as file:
            torch.save(model, file)

    config_file_store = os.path.join(dir_name, "config.yml")
    shutil.copy(args.config_file, config_file_store)
    discord.send_directory(dir_name)
