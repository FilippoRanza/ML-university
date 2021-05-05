#! /usr/bin/python

from argparse import ArgumentParser

import pandas as pd
import numpy as np
from utils import load_dataset, export_data

def collapse(orig_dataset):
    output = pd.DataFrame()
    matrix = orig_dataset.values
    rows, _ = matrix.shape
    vec = np.zeros(rows, dtype=np.uint32)
    for i, row in enumerate(matrix):
        loc, = np.where(row == 1)
        if loc.size == 0:
            index = 0
        else:
            index = loc[0] + 1
        vec[i] = index

    output['class'] = vec
    return output

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    in_dataset = load_dataset(args.input)
    out_dataset = collapse(in_dataset)
    export_data(out_dataset, args.output)

if __name__ == '__main__':
    main()











