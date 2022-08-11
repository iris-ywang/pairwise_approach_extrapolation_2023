from split_data import load_and_check_data
from perform_base_case import run_base_models
from perform_stacking import run_stacking
import os


def load_datasets():
    """
    Create a list of directories for all the qsar datasets to be evaluated
    :return:
    list of strings
    """
    filename_lst = []
    # TODO: may need to change the way of getting parent directory if this duoes not work on windows
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
    # If dataset passes the criteria, then it gives a dict of fold number and their corresponding pre-processed data
    data = load_and_check_data(file)
    if data is None:
        continue

    meta_data = run_base_models(data)  # a dict: key = fold number, values = (x_meta, y_meta)
    metrics = run_stacking(data, meta_data)  # np array: shape = (number_of_fold, number_of_base+1, number_of_metric)
    all_metrics.append(metrics)
