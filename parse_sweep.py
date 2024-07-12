"""
Input: a subdirectory containing the logs from various experiments
Output: a csv file with loss values, peak memory usage, throughout from each experiment
"""

import csv
import os
import re

import fire

OUTPUT_FOLDER = '/home/vasiliy/local/tmp/torchtitan_outputs'

# example: 
# [rank0]:[INFO     | root               ]: step: 10  loss:  7.8774  memory:  0.44GiB(0.47%)  tps: 997,458  mfu: 1.50%
# note that number of spaces between terms can vary
regex = r"- step:[ ]+([\d]+).*loss:[ ]+([\d\.]+).*memory:[ ]+([\d\.]+)GiB.*tps: ([\d\,]+).*mfu.*"

def log_to_maybe_data(line):
    res = re.search(regex, line)
    if res is not None:
        step, loss, memory_gib, wps = res.group(1), res.group(2), res.group(3), res.group(4)
        return int(step), float(loss), float(memory_gib), int(wps.replace(',', ''))
    else:
        return None

def run(
    subfolder_prefix: str,
    results_filename: str,
):
    subfolder_prefix = str(subfolder_prefix)

    results = [['experiment', 'step', 'loss', 'memory_gib', 'tps']]

    for entry in os.scandir(OUTPUT_FOLDER):
        if entry.is_dir() and subfolder_prefix in entry.path:
            print(entry)
            log_fname = f"{entry.path}/logs.txt"
            short_path = entry.path.replace(f"{OUTPUT_FOLDER}/", '')
            
            with open(log_fname, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    res = log_to_maybe_data(l)
                    if res is not None:
                        print(l.strip('\n'))
                        print(res)
                        results.append([short_path, *res])

    with open(results_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print('done')

if __name__ == '__main__':
    fire.Fire(run)
