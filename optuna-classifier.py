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
    BALANCED = data.get("BALANCED_ACCURACY", False)
    WEIGHTS = data.get("WEIGHTS", False)
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


def get_test_result(model):
    cpu_dev = torch.device("cpu")
    model.to(cpu_dev)
    model.eval()
    x, y_true = load_datasets(TEST_FEATURE, TEST_TARGET)
    x = torch.from_numpy(x)
    with torch.no_grad():
        y_pred = model(x)
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(axis=1)
    return y_true, y_pred


def get_score(model):
    y_true, y_pred = get_test_result(model)
    repo = metrics.classification_report(y_true, y_pred)
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    hinge = metrics.hinge_loss(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    b_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    result = f"Report:\n{repo}\nConfusion Matrix:\n{conf_mat}\nHinge Loss: {hinge}\nROC AUC: {auc}\nBalanced Accuracy: {b_acc}\n"

    return result


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 25)
    layers = []

    in_features = FEATURES
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 192)
        layers.append(nn.Linear(in_features, out_features))
        nn_linear_name = trial.suggest_categorical(
            f"non_linear-{i}",
            [
                "ReLU",
                "LogSigmoid",
                "Sigmoid",
                "Tanh",
            ],
        )
        nn_linear = getattr(nn, nn_linear_name)
        layers.append(nn_linear())

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def suggest_weights(trial, count):
    weights = [trial.suggest_float(f"weight_{i}", 0.5, 100) for i in range(count)]
    weights = np.array(weights, dtype=np.float32)
    return weights


class AccuracyScore:
    def __init__(self):
        self.count = 0

    def get_accuracy(self, item_count):
        return self.count / item_count

    def add_score(self, y_true, y_pred):
        y_true = y_true.to(DEVICE)
        y_pred = y_pred.to(DEVICE)
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.count += pred.eq(y_true.view_as(pred)).sum().item()


class AbstractAccuracyScore:
    def __init__(self):
        self.target_values = np.zeros(0)
        self.result_values = np.zeros(0)

    def get_accuracy(self, _item_count):
        raise NotImplemented

    def add_score(self, y_true, y_pred):
        pred = y_pred.argmax(dim=1, keepdim=True)
        pred = pred.cpu().numpy()
        self.target_values = np.concatenate((self.target_values, y_true.numpy()))
        self.result_values = np.concatenate((self.result_values, pred[:, 0]))

class BalancedAccuracyScore(AbstractAccuracyScore):
    def __init__(self):
        super(BalancedAccuracyScore, self).__init__()

    def get_accuracy(self, _item_count):
        return metrics.balanced_accuracy_score(self.target_values, self.result_values)



class BalancedLossScore(AbstractAccuracyScore):
    def __init__(self):
        super(BalancedLossScore, self).__init__()

    def get_accuracy(self, _item_count):
        conf_mat = metrics.confusion_matrix(self.target_values, self.result_values)
        output = np.zeros(len(conf_mat), dtype=np.float32)
        for i, row in enumerate(conf_mat):
            den = np.sum(row)
            output[i] = 1 - (row[i] / den)
        return np.sum(output)
        

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
    def __init__(self, weights):
        self.models = {}
        self.weights = weights

    def get_best(self):
        trial = 0
        if BALANCED == 'loss': 
            accuracy = CLASSES
        else:
            accuracy = 0
        model = None
        for k, (m, a) in self.models.items():
            if BALANCED == 'loss':
                cond = a < accuracy
            else: 
                cond = a > accuracy
         
            if cond:
                accuracy = a
                trial = k
                model = m
        return (trial, model)

    def automatic_weights(self):
        _, best = self.get_best()
        if best:
            output = self._compute_automatic_weights(best)
        else: 
            output = np.ones(CLASSES, dtype=np.float32)
        
        return torch.from_numpy(output).to(DEVICE)

    def _compute_automatic_weights(self, best):
        y_true, y_pred = get_test_result(best)
        conf_mat = metrics.confusion_matrix(y_true, y_pred)
        output = np.zeros(len(conf_mat), dtype=np.float32)
        for i, row in enumerate(conf_mat):
            den = np.sum(row)
            output[i] = 2 - (row[i] / den) 
        output *= 10

        delta = (max(output) - min(output)) / min(output)
        
        if delta > .4:
            i = np.argmax(output)
            output[i] *= 2

        return output

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

        if self.weights:
            weight = trial.suggest_categorical(
                "weights", [None,  "automatic"]
            )
            if weight:
                weight = {
                    "inverse": proportional_weights,
                    "automatic": self.automatic_weights(),
                }[weight]
        else:
            weight = None

        loss_function_name = trial.suggest_categorical(
            "loss function", ["nll_loss", "cross_entropy"]
        )
        loss_function = getattr(F, loss_function_name)

        # Training of the model.
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target, weight=weight)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            acc_score = {"balanced": BalancedAccuracyScore, "default": AccuracyScore, "loss": BalancedLossScore}[BALANCED]
            acc_score = acc_score()

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    data = data.to(DEVICE)
                    output = model(data)
                    acc_score.add_score(target, output)

            accuracy = acc_score.get_accuracy(len(valid_loader.dataset))

            trial.report(accuracy, epoch)
            self.models[trial.number] = (model, accuracy)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy


if __name__ == "__main__":
    discord.send_message("Start OPTUNA study")
    if BALANCED == 'loss':
        direction = "minimize"
    else:
        direction="maximize"

    study = optuna.create_study(
        direction=direction,
        study_name=STUDY_NAME,
        storage=DB_FILE,
        load_if_exists=True,
    )

    obj = Objective(WEIGHTS)
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
