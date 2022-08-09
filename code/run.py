from split_data import load_and_check_data
from perform_base_case import run_base_models
from perform_stacking import run_stacking

import os


def load_datasets():
    filename_lst = []
    directory = os.getcwd() + '/input/qsar_data_unsorted'

    for root, dirs, files in os.walk(directory):
        for each_file in files:
            if each_file.endswith(".csv"):
                f = open(os.path.join(root, each_file), 'r')
                filename_lst.append(os.path.join(root, each_file))
                f.close()
    return filename_lst


filename_list = load_datasets()
all_metrics = []
for file in filename_list:
    data = load_and_check_data(file)
    if data is None:
        continue
    meta_data = run_base_models(data)
    metrics = run_stacking(data, meta_data)
    all_metrics.append(metrics)
