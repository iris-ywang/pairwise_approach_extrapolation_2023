import os
import numpy as np
import pandas as pd
import time

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
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
    chembl_info = pd.read_csv("input//chembl_datasets_info.csv")
    try:
        existing_results = np.load("PA_all_data_hpc.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("temporary_dataset_count.npy")
    except:
        np.save("temporary_dataset_count.npy", [0])

    count = 0
    for file in range(len(chembl_info)):
        # if len(all_metrics) > 100: break
        # if chembl_info["Repetition Rate"][file] >= 0.50: continue
        # if chembl_info["N(sample)"][file] > 1000 or chembl_info["N(sample)"][file] < 10: continue
        # If dataset passes the above criteria, then it gives a dict of fold number and their corresponding

        # HPC filtering: starting from dataset 251:
        if file < 251: continue
        # and in this way, I think it is missing dataset 0
        count += 1
        if count <= existing_count: continue

        print("On Dataset No.", count, ", ", chembl_info["File name"][file])

        # TODO: may need to change the way of getting parent directory if this does not work on windows
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + chembl_info["File name"][file], shuffle_state=1)
        data = generate_train_test_sets_ids(train_test, fold=10)

        # print("Running models...")
        start = time.time()
        metrics = run_model(data, current_dataset_count=count)
        # print(":::Time used: ", time.time() - start, "\n")

        all_metrics.append(metrics)
        np.save("PA_all_data_hpc.npy", np.array(all_metrics))

