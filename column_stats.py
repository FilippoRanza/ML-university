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

def get_column_stats(dataset, output: pd.DataFrame):
    stats = {
        'average': [],
        'variance': [],
        'min': [],
        'max': [],
        'median':[]
    }

    methods = {
        'average': pd.DataFrame.mean,
        'variance': pd.DataFrame.var,
        'min': pd.DataFrame.min,
        'max': pd.DataFrame.max,
        'median': pd.DataFrame.median,
    }

    for key, col in dataset.items():
        if col.dtype == np.float64 or col.dtype == np.int64:
            for k, m in methods.items():
                try:
                    stats[k].append(m(col))
                except:
                    print(k)
                    raise ValueError
        else:
            for vec in stats.values():
                vec.append(None)
            
        

    for k, v in stats.items():
        output[k] = v

    
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

    try:
        patient_ids = data['Patient ID']
        duplicates = find_duplicates(patient_ids)
        if len(duplicates):
            print("Duplicated Patient ID")
            exit(1)
    except:
        pass

    output = init_output_dataframe(data)
    find_incomplete_column(data, output)
    get_column_stats(data, output)
    output.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
