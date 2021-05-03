#! /usr/bin/python

import json
import sys

from matplotlib import pyplot as plt

in_file = sys.argv[1]
try: 
    out_file = sys.argv[2]
except:
    out_file = None

with open(in_file) as file:
    data = json.load(file)

x = []
y = []
for k, v in data.items():
    x.append(k)
    y.append(len(v))

fig, ax = plt.subplots()
ax.bar(x, y)
ax.set_xticklabels([f"${i:.3}$" for i in x])
ax.set_xlabel("$Usage\ Ratio$")
ax.set_ylabel("$Quantity$")
ax.set_title(r'$Usage\ Ratio = \frac{|Non\ Null\ Items|}{|Items|}$')


if out_file:
    plt.savefig(out_file)
else:
    plt.show()



