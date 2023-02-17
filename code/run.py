import os
import numpy as np
from datetime import datetime

import pandas as pd
import time
import warnings
import multiprocessing

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


def run_datasets_in_parallel(file):
    directory_connection = "/input/processed_tml_datasets_100/"
    all_file_names = os.listdir(os.getcwd() + directory_connection)
    file_count = file + 1
    # TODO: may need to change the way of getting parent directory if this does not work on windows
    filename = all_file_names[file]
    print("On Dataset No.", file_count, ", ", filename)

    with open(os.getcwd() + "/extrapolation_svm/"+'dataset_running_order_tml_svm_100.txt', 'a') as f:
        f.write("\n"+"Running Dataset No."+str(file_count)+filename + "\n")

    train_test = dataset(os.getcwd() + directory_connection + filename, shuffle_state=1)
    print("Generating datasets...")
    start = time.time()
    data = generate_train_test_sets_ids(train_test, fold=10)
    print(":::Time used: ", time.time() - start)

    print("Running models...")
    start = time.time()
    metrics = run_model(data, current_filename=filename, percentage_of_top_samples=0.1)
    print(":::Time used: ", time.time() - start, "\n")

    np.save(os.getcwd() + "/extrapolation_svm/" + "extrapolation_10fcv_tml_svm_100"+str(filename)+".npy", metrics)
    with open(os.getcwd() + "/extrapolation_svm/"+'dataset_running_order_tml_svm_100.txt', 'a') as f:
        f.write("\n"+"Finished Dataset No."+str(file_count)+filename + "\n")


def count_finished_datasets(all_file_names):
    existing_count = 0
    for file in range(len(all_file_names)):
        filename = all_file_names[file]
        try:
            _ = np.load(os.getcwd()
                        + "/extrapolation_svm/"
                        + "extrapolation_10fcv_tml_svm_100" + str(filename) + ".npy")
            existing_count += 1
        except FileNotFoundError:
            return existing_count


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    connection = "/input/processed_tml_datasets_100/"
    list_of_file_names = os.listdir(os.getcwd() + connection)

    existing_count = count_finished_datasets(list_of_file_names)

    with open(os.getcwd() + "/extrapolation_svm/"+'dataset_running_order_tml_svm_100.txt', 'a') as f:
        f.write("\n"+str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")

    with multiprocessing.Pool() as executor:
        executor.map(run_datasets_in_parallel, range(existing_count, len(list_of_file_names)), 1)



