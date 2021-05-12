#! /usr/bin/python

from argparse import ArgumentParser
import os

from sklearn import model_selection
import pandas as pd

from utils import load_dataset, export_dataset


def parse_args():
    parser = ArgumentParser(
        description="Split feature and target set in training and test set, train and test set file are named automatically from feature and target file"
    )
    parser.add_argument(
        "-f", "--feature-set", help="Specify feature set file", required=True
    )
    parser.add_argument(
        "-t", "--target-set", help="Specify target set file", required=True
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=0.1,
        help="Specify test set size [0.0 - 1.0]",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="shuffle dataset before split",
    )
    return parser.parse_args()


def split_set(feature_set, target_set, test_size, shuffle):
    feature_set = load_dataset(feature_set).values
    target_set = load_dataset(target_set).values
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        feature_set, target_set, test_size=test_size, shuffle=shuffle
    )

    return (x_train, x_test), (y_train, y_test)

def export_set(data, file_name):
    dataframe = pd.DataFrame(data)
    export_dataset(dataframe, file_name)    

def export_files(train_name, test_name, train_set, test_set):
    export_set(train_set, train_name)
    export_set(test_set, test_name)


def get_names(file_name):
    base_name = os.path.basename(file_name)
    dir_name = os.path.dirname(file_name)

    train_name = f"train-{base_name}"
    train_name = os.path.join(dir_name, train_name)
    test_name = f"test-{base_name}"
    test_name = os.path.join(dir_name, test_name)

    return train_name, test_name


def save_set(orig_name, sets):
    train_set, test_set = sets
    train_name, test_name = get_names(orig_name)
    export_files(train_name, test_name, train_set, test_set)


def main():
    args = parse_args()
    feature, target = split_set(
        args.feature_set, args.target_set, args.size, args.shuffle
    )
    save_set(args.feature_set, feature)
    save_set(args.target_set, target)


if __name__ == "__main__":
    main()
