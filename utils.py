#! /usr/bin/python

import pathlib
import pandas as pd


def load_dataset(file_name):
    path = pathlib.Path(file_name)
    ext = path.suffix
    reader = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel
    }
    try:
        output = reader[ext](path)
    except KeyError:
        raise ValueError(f"Unkwnon extension: {ext}")

    return output
