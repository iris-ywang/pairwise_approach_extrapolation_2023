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
        existing_results = np.load("PA_all_data.npy")
    except:
        np.save("PA_all_data.npy", np.zeros((1, 10, 3, 18)))

    for file in range(len(chembl_info)):
        existing_results = np.load("PA_all_data.npy")
        count = len(existing_results)
        # if len(all_metrics) > 100: break
        # if chembl_info["Repetition Rate"][file] >= 0.50: continue
        # if chembl_info["N(sample)"][file] > 1000 or chembl_info["N(sample)"][file] < 10: continue
        # If dataset passes the above criteria, then it gives a dict of fold number and their corresponding

        print("On Dataset No." + str(count))

        # TODO: may need to change the way of getting parent directory if this does not work on windows
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + chembl_info["File name"][file], shuffle_state=1)
        data = generate_train_test_sets_ids(train_test, fold=10)

        print("Running models...")
        start = time.time()
        metrics = run_model(data)
        print(":::Time used: ", time.time() - start, "\n")

        new_results = np.concatenate((existing_results, metrics), axis=0)
        np.save("PA_all_data.npy", np.array(new_results))

