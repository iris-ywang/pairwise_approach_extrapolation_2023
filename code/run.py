import os
import numpy as np
import pandas as pd
import time
import warnings

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_with_K_fold_forward
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
    all_metrics = []


    for file in range(len(chembl_info)):
        if chembl_info["Repetition Rate"][file] > 0.1: continue
        if chembl_info["N(sample)"][file] > 300 or chembl_info["N(sample)"][file] < 90: continue

        # TODO: may need to change the way of getting parent directory if this does not work on windows
        print("On Dataset No.", count, ", ", chembl_info["File name"][file])
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + chembl_info["File name"][file], shuffle_state=1)
        sorted_train_test = train_test[train_test[:,0].argsort()]
        metrics = generate_train_test_sets_with_K_fold_forward(sorted_train_test, folds=10)
        # metrics = run_model(data, percentage_of_top_samples=0.1)

        all_metrics.append(metrics)
        np.save("extrapolation_validation_xiong_run1.npy", np.array(all_metrics))

