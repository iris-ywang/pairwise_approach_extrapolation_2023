import os
import numpy as np
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
    chembl_info = pd.read_csv("input//chembl_datasets_info.csv").sort_values(by=["N(sample)"])
    file_count = file + 1
    # TODO: may need to change the way of getting parent directory if this does not work on windows
    filename = chembl_info.iloc[file]["File name"]
    print("On Dataset No.", file_count, ", ", filename)

    with open(os.getcwd() + "/extrapolation_svm/"+'dataset_svm_running_order.txt', 'a') as f:
        f.write("Running Dataset No."+str(file_count)+filename + "\n")

    connection = "/input/qsar_data_unsorted/"
    train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)
    print("Generating datasets...")
    start = time.time()
    data = generate_train_test_sets_ids(train_test, fold=10)
    print(":::Time used: ", time.time() - start)

    print("Running models...")
    start = time.time()
    metrics = run_model(data, current_filename=filename, percentage_of_top_samples=0.1)
    print(":::Time used: ", time.time() - start, "\n")

    np.save(os.getcwd() + "/extrapolation_svm/" + "extrapolation_svm_kfold_cv_all_data_hpc_"+str(filename)+".npy", metrics)
    with open('dataset_svm_running_order.txt', 'a') as f:
        f.write("\n"+"Finished Dataset No."+str(file_count)+filename + "\n")


def count_finished_datasets(sorted_chembl_info):
    existing_count = 0
    for file in range(len(sorted_chembl_info)):
        filename = sorted_chembl_info.iloc[file]["File name"]
        try:
            _ = np.load(os.getcwd() + "/extrapolation_svm/" + "extrapolation_svm_kfold_cv_all_data_hpc_"+str(filename)+".npy")
            existing_count += 1
        except FileNotFoundError:
            return existing_count


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    chembl_info = pd.read_csv("input//chembl_datasets_info.csv").sort_values(by=["N(sample)"])
    existing_count = count_finished_datasets(chembl_info)

    with multiprocessing.Pool() as executor:
        executor.map(run_datasets_in_parallel, range(existing_count, len(chembl_info)),1)



