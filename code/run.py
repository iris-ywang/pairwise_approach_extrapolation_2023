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

    chembl_info = pd.read_csv("input//chembl_datasets_info.csv")

    try:
        existing_results = np.load("extrapolation_active_learning_run1.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        all_metrics = []

    count = 0
    for file in range(len(chembl_info)):
        if chembl_info["Repetition Rate"][file] > 0.5: continue
        if chembl_info["N(sample)"][file] > 1800 or chembl_info["N(sample)"][file] < 1000: continue
        # if chembl_info["File name"][file] in list_of_files: continue
        count += 1
        if count <= existing_count: continue

        # TODO: may need to change the way of getting parent directory if this does not work on windows
        print("On Dataset No.", count, ", ", chembl_info["File name"][file])
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + chembl_info["File name"][file], shuffle_state=1)

        metrics = generate_train_test_sets_with_increasing_train_size(train_test, current_dataset_count=count)

        all_metrics.append(metrics)
        np.save("extrapolation_active_learning_run1.npy", np.array(all_metrics))

