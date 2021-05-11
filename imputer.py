#! /usr/bin/python

from argparse import ArgumentParser

import pandas as pd
from sklearn import impute, preprocessing

from utils import load_dataset, export_dataset

parser = ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
parser.add_argument('--scale', default=False, action='store_true')
parser.add_argument('--iterative', default=False, action='store_true')

args = parser.parse_args()

data = load_dataset(args.input)

if args.iterative:
    from sklearn.experimental import enable_iterative_imputer 
    imp = impute.IterativeImputer(max_iter=15000, initial_strategy="most_frequent", imputation_order="descending")
else:
    imp = impute.SimpleImputer(strategy="most_frequent")

imp.fit(data.values)

output_data = imp.transform(data.values)
if args.scale:
    scaler = preprocessing.StandardScaler()
    scaler.fit(output_data)
    output_data = scaler.transform(output_data)

output_frame = pd.DataFrame(data=output_data, columns=data.keys())


export_dataset(output_frame, args.output)

