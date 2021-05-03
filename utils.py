#! /usr/bin/python

import pathlib
import pandas as pd

import json
import yaml


def __get_extension__(file_name):
    path = pathlib.Path(file_name)
    ext = path.suffix
    return ext

def load_dataset(file_name, transpose=False):
    reader = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel
    }
    ext = __get_extension__(file_name)
    try:
        output = reader[ext](file_name)
    except KeyError:
        raise ValueError(f"Unkwnon extension: {ext}")

    if transpose:
        output = output.transpose()

    return output


def export_dataset(dataset: pd.DataFrame, file_name: str):
    ext = __get_extension__(file_name)
    if ext == '.csv':
        dataset.to_csv(file_name, index=False)
    elif ext == '.xlsx':
        dataset.to_excel(file_name, index=False)
    else:
        raise ValueError(f"Unknown extension: {ext}")


def export_json(data, file_name: str):
    with open(file_name, "w") as output:
        json.dump(data, output)

def export_yaml(data, file_name: str):
    with open(file_name, "w") as output:
        yaml.dump(data, file_name)


def __marshal_data__(data, file_name):
    ext = __get_extension__(file_name)
    writers = {
        '.json': json.dump,
        '.yml': yaml.dump
    }
    try:
        with open(file_name, "w") as file:
            writers[ext](data, file)
    except KeyError:
        raise ValueError(f"Unknown extensin: {ext}")

def export_data(data, file_name: str):
    if type(data) == pd.DataFrame:
        export_dataset(data, file_name)
    else:
        __marshal_data__(data, file_name)
    

