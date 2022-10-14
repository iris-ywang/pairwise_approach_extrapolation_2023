import os
import numpy as np
import pandas as pd
import time
import warnings

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_with_increasing_train_size
from build_model import run_model


def load_datasets():
    """
    Create a list of directories for all the qsar datasets to be evaluated
    :return:
    list of strings
    """
    filename_lst = []
    # TODO: may need to change the way of getting parent directory if this does not work on windows
    directory = os.getcwd() + '/input/qsar_data_unsorted'

    for root, dirs, files in os.walk(directory):
        for each_file in files:
            if each_file.endswith(".csv"):
                f = open(os.path.join(root, each_file), 'r')
                filename_lst.append(os.path.join(root, each_file))
                f.close()
    return filename_lst


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # chembl_info = pd.read_csv("input//chembl_datasets_info.csv")
    all_metrics = []

    number_of_existing_results = 101
    count = 0
    list_of_files = ["data_CHEMBL229.csv", "data_CHEMBL4805.csv", "data_CHEMBL268.csv",
                     "data_CHEMBL283.csv"]
    for file in list_of_files:
        #  list_of_files_done: [data_CHEMBL3286(size, 1002; repetition rate: 0.04),
        #  "data_CHEMBL5071.csv" size 1002, repetition 0.019,]
        # a list of low repetition rate


        # TODO: may need to change the way of getting parent directory if this does not work on windows
        print("On Dataset No.", count, ", ", file)
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + file, shuffle_state=1)
        metrics = generate_train_test_sets_with_increasing_train_size(train_test, step_size=0.03)
        # metrics = run_model(data, percentage_of_top_samples=0.1)

        all_metrics.append(metrics)
        np.save("extrapolation_increase_train_size_run4.npy", np.array(all_metrics))

