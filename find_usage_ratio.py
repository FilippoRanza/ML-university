#! /usr/bin/python

from argparse import ArgumentParser
import json
import csv

import numpy as np
import pandas as pd

from utils import load_dataset

def split_by_ratio(feature_usage_ratio: list, count=11):
    indexes = np.linspace(0, 1, count)
    output = {i: [] for i in indexes}
    for name, ratio in feature_usage_ratio:
        for i in indexes:
            if ratio <= i:
                output[i].append(name)
                break
        else:
            output[1.0].append(name)
    return {k: v for k, v in output.items() if v}


def get_usage_ratio(dataset):
    def __inner__():
        for feature, values in dataset.items():
            ratio = values.count() / len(values)
            yield feature, ratio
    return list(__inner__())


def output_json(data, file_name):
    with open(file_name, "w") as output:
        json.dump(data, output)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('output')
    parser.add_argument('-t', '--transpose', default=False, action='store_true')
    parser.add_argument('-g', '--group', default=True, action='store_false')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.input_dataset, args.transpose)
    usage_ratio = get_usage_ratio(dataset)
    if args.group:
        split_features = split_by_ratio(usage_ratio)
        output_json(split_features, args.output)
    else:
        with open(args.output, "w") as file:
            writer = csv.writer(file)
            for key, value in usage_ratio:
                writer.writerow((key, value))
            


if __name__ == '__main__':
    main()















