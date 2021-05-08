#! /usr/bin/python

from argparse import ArgumentParser
import sys

def load_log(log_file):
    logs = []
    with open(log_file) as file:
        enable_read = False
        for line in file:
            line = line.strip()
            if enable_read and line:
                logs.append(line)

            if line == 'Grid Search statistics:':
                enable_read = True

    return logs

def sort_logs(logs: list):
    logs.sort(key=lambda x: -float(x.split()[0]))
    return logs

def remove_nan(logs: list):
    output = []
    for log in logs:
        tokens = log.split()
        if tokens[0] != 'nan':
            output.append(log)
    return output

def output_logs(logs, file):
    for log in logs:
        print(log, file=file)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument('-r', '--remove-nan', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logs = load_log(args.input)
    if args.remove_nan:
        logs = remove_nan(logs)
    
    logs = sort_logs(logs)
 
    if args.output:
        with open(args.output, "w") as file:
            output_logs(logs, file)
    else:
        output_logs(logs, sys.stdout)




if __name__ == '__main__':
    main()






