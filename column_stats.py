#! /usr/bin/python

from argparse import ArgumentParser

import pandas as pd
import numpy as np

from utils import load_dataset


def find_duplicates(pat_id):
    count = {}
    for i in pat_id:
        try:
            count[i] += 1
        except KeyError:
            count[i] = 1
    output = {k: v for k, v in count.items() if v > 1}
    return output

def find_incomplete_column(dataset, output: pd.DataFrame):
    tmp = []
    for key, col in dataset.items():
        ratio = col.count() / len(col)
        tmp.append(ratio)

    output['ratio'] = tmp


    
def init_output_dataframe(dataset):
    output = pd.DataFrame()
    features = dataset.columns
    output['feature_name'] = features
    return output




def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('output_file')
    return parser.parse_args()



def main():
    args = parse_args()
    data = load_dataset(args.dataset)


    patient_ids = data['Patient ID']
    duplicates = find_duplicates(patient_ids)
    if len(duplicates):
        print("Duplicated Patient ID")
        exit(1)

    output = init_output_dataframe(data)
    find_incomplete_column(data, output)
    output.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
