import os
import numpy as np
import pandas as pd
import time
import warnings

from pa_basics.import_chembl_data import uniform_features, duplicated_features
from split_data import train_test_sets_splits
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


def gene_data_combine_train_test(train_filename, test_filename, number_of_train_to_use):

    train_all = pd.read_csv(train_filename, index_col=0).iloc[:number_of_train_to_use, :]
    test_all = pd.read_csv(test_filename, index_col=0)

    train_test = pd.concat([train_all, test_all], ignore_index=True)
    cols = train_test.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    train_test = np.array(train_test[cols])

    filter1 = uniform_features(train_test)
    filter2 = duplicated_features(filter1)

    return filter2


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    train_filename = os.getcwd() + "/input" + '/ABCC5_base_train.csv'
    test_filename = os.getcwd() + "/input" + '/ABCC5_base_test.csv'

    metrics_all = []
    for number_of_train_to_use in range(450, 810, 10):
        print("Size:", number_of_train_to_use)
        train_test = gene_data_combine_train_test(train_filename, test_filename, number_of_train_to_use)
        data = train_test_sets_splits(train_test, number_of_train_to_use)
        print("Running models...")
        start = time.time()
        metrics = run_model(data, percentage_of_top_samples=0.1)
        print(":::Time used: ", time.time() - start, "\n")
        metrics_all.append(metrics[0])
        print(np.nanmean(metrics[0], axis=0))
        np.save("ABCC5_base_train_results_450.npy", np.array(metrics_all))


