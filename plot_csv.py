#! /usr/bin/python

from argparse import ArgumentParser
import csv

from matplotlib import pyplot as plt


def parse_file(file_name, auto):
    x_list = []
    y_list = []
    with open(file_name) as file:
        reader = csv.reader(file)
        for i, entry in enumerate(reader):
            x, y = entry
            if auto:
                x_list.append(i)
            else:
                x = float(x)
                x_list.append(x)     
            y = float(y)
            y_list.append(y)

    return x_list, y_list  


def parse_args():
    parser = ArgumentParser(description="plot data from a TWO columns CSV file: first column is X, second column is Y")
    parser.add_argument('-i', '--input', help="input CSV file", required=True)
    parser.add_argument('-o', '--output', help="output file")
    parser.add_argument('-t', '--title', help="set plot title")
    parser.add_argument('-a', '--auto', help="Automatically compute X value using entry index, ignore first column", default=False, action="store_true")
    parser.add_argument('-l', '--legend', help="set legent")
    parser.add_argument('--x-label', help="set X label")
    parser.add_argument('--y-label', help="set Y label")
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = parse_file(args.input, args.auto)
    fig, ax = plt.subplots()
    ax.plot(x, y)

    if args.title:
        ax.set_title(args.title)

    if args.legend:
        ax.legend([args.legend])

    if args.x_label:
        ax.set_xlabel(args.x_label)

    if args.y_label:
        ax.set_ylabel(args.y_label)

    fig.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    main()



