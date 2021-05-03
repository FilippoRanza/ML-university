#! /usr/bin/python

from argparse import ArgumentParser

import pandas as pd
import yaml

from utils import load_dataset, export_data


def is_number(string: str) :
    try: 
        _ = float(string)
    except ValueError:
        return False
    return True
    

class ValueMapper:
    def __init__(self):
        self.mapping = {}
        self.current_counter = 0

    def get_mapping(self, column, value):
        if is_number(value) or column == 'Patient ID':
            return value

        try:
            col_mapper = self.mapping[column]
            try:
                output = col_mapper[value]
            except KeyError:
                output = self.current_counter
                col_mapper[value] = self.current_counter
                self.current_counter += 1
        except KeyError:
            self.current_counter = 0
            output = self.current_counter
            self.mapping[column] = {value: self.current_counter}
            self.current_counter += 1

        return output


def map_values(col_id, values, mapper: ValueMapper):
    def __inner__():
        for v in values:
            if v:
                yield mapper.get_mapping(col_id, v)
            else:
                yield v
    return list(__inner__())


def str_to_number(dataset):
    output = pd.DataFrame()
    mapper = ValueMapper()
    for key, values in dataset.items():
        output[key] = map_values(key, values, mapper)
    return output, mapper

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('name_map')
    parser.add_argument('output_dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.input_dataset)
    out_dataset, mapper = str_to_number(dataset)
    export_data(out_dataset, args.output_dataset)
    export_data(mapper.mapping, args.name_map)


if __name__ == '__main__':
    main()
