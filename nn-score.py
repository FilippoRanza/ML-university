from argparse import ArgumentParser
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics


import numpy as np

from utils import load_dataset, time_stamp, DiscordFrontEnd

device = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device(device)
DIR = os.getcwd()



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


def get_score(model, feature, target):
    cpu_dev = torch.device("cpu")
    model.to(cpu_dev)
    model.eval()
    x, y_true = load_datasets(feature, target)
    x = torch.from_numpy(x)
    with torch.no_grad():
        y_pred = model(x)
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(axis=1)

    repo = metrics.classification_report(y_true, y_pred)
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    hinge = metrics.hinge_loss(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    b_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    result = f"Report:\n{repo}\nConfusion Matrix:\n{conf_mat}\nHinge Loss: {hinge}\nROC AUC: {auc}\nBalanced Accuracy: {b_acc}\n"

    return result

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--feature-set', required=True)
    parser.add_argument('-t', '--target-set', required=True)
    parser.add_argument('-m', '--model', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.model, "rb") as f:
        model = torch.load(f)

    score = get_score(model, args.feature_set, args.target_set)

    print(score)
    