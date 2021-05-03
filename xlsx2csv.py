#! /usr/bin/python

import sys
import pandas

input_file = sys.argv[1]
output_file = sys.argv[2]

data = pandas.read_excel(input_file)
data.to_csv(output_file, index=False)
