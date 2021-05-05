#! /usr/bin/python

import pandas as pd
from argparse import ArgumentParser

from utils import  load_dataset, export_data

def get_feauture_key(ratio: pd.DataFrame, min_ratio: float, ignore_list: list):
    sort_items = ratio.sort_values(by='ratio', ascending=False).drop(ignore_list)
    output = []
    for val in sort_items.values:
        if val[1] < min_ratio:
            break
        output.append(val[0])
    return output

def make_new_dataset(orig_dataset, ratio, min_ratio, ignore_list):
    keys = get_feauture_key(ratio, min_ratio, ignore_list)
    output = pd.DataFrame()
    for key in keys:
        output[key] = orig_dataset[key]
    return output

def filter_empty_items(dataset, min_ur):
    drop_rows = []
    for index, row in dataset.iterrows():
        ratio = row.count() / len(row)
        if ratio < min_ur:
            drop_rows.append(index)

    return dataset.drop(drop_rows)


def filter_all_equal(dataset):
    drop_rows = []
    for i, row in dataset.iterrows():
        min_val = row[1:].min()
        max_val = row[1:].max()
        if min_val == max_val:
            drop_rows.append(i)

    return dataset.drop(drop_rows)

def make_target_set(orig_dataset: pd.DataFrame, new_dataset: pd.DataFrame, target_col: list):
    output = pd.DataFrame()
    for col_key in orig_dataset.take(target_col, axis=1):
        indexes = new_dataset.index
        col = orig_dataset[col_key]
        target = col.take(indexes)
        output[col_key] = target
    return output
 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-s', '--stat', required=True)
    parser.add_argument('-m', '--min-ratio', type=float, required=True)
    parser.add_argument('-i', '--ignore-list', nargs='+', type=int, required=True)
    parser.add_argument('-u', '--min-ur', type=float, required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-t', '--target')
    parser.add_argument('-c', '--target-column', nargs='+', type=int)
    return parser.parse_args()



def main():
    args = parse_args()
    orig_dataset = load_dataset(args.dataset)
    ratio = load_dataset(args.stat)
    new_dataset = make_new_dataset(orig_dataset, ratio, args.min_ratio, args.ignore_list)
    new_dataset = filter_empty_items(new_dataset, args.min_ur)
    new_dataset = filter_all_equal(new_dataset)
    export_data(new_dataset, args.output)
    if args.target:
        target_set = make_target_set(orig_dataset, new_dataset, args.target_column)
        export_data(target_set, args.target)

if __name__ == '__main__':
    main()




