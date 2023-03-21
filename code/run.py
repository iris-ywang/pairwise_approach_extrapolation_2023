import os
import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_ids
from build_model import run_model


def process_datasets(create_size=100):
    new_files = np.load("tml_gene_names.npy")
    for gene_name in new_files:
        train = pd.read_csv(gene_name + "_base_train.csv", index_col=0).sample(n=int(0.7*create_size))
        test = pd.read_csv(gene_name + "_base_test.csv", index_col=0).sample(n=int(0.3*create_size))
        dataset = pd.concat([train, test], ignore_index=True)
        cols = dataset.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dataset = dataset[cols]
        dataset.to_csv(os.getcwd() + "/processed_tml_datasets/" + gene_name + "_"+str(create_size)+".csv", index=False)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    connection = "/input/processed_tml_datasets_100/"
    list_of_file_names = os.listdir(os.getcwd() + connection)
    all_metrics = []

    try:
        existing_results = np.load("extrapolation_10fcv_tml_ridge_100.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except:
        existing_results = None
        existing_count = 0
        all_metrics = []

    try:
        _ = np.load("tml_temporary_dataset_ridge_count_100.npy")
    except:
        np.save("tml_temporary_dataset_ridge_count_100.npy", [0])

    with open('dataset_running_order_tml_ridge_100.txt', 'a') as f:
        f.write("\n"+str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")

    count = 0
    for file in range(len(list_of_file_names)):
        # if len(all_metrics) > 100: break
        # if chembl_info["Repetition Rate"][file] > 0.15: continue
        # if chembl_info["N(sample)"][file] > 300 or chembl_info["N(sample)"][file] < 90: continue
        # If dataset passes the above criteria, then it gives a dict of fold number and their corresponding
        # pre-processed data

        count += 1
        if count <= existing_count:
            continue
        # TODO: may need to change the way of getting parent directory if this does not work on windows
        filename = list_of_file_names[file]
        if ".csv" not in filename:
            continue
        print("On Dataset No.", count, ", ", filename)

        with open('dataset_running_order_tml_ridge_100.txt', 'a') as f:
            f.write(filename)

        train_test = dataset(os.getcwd() + connection + filename, shuffle_state=1)
        print("Generating datasets...")
        start = time.time()
        data = generate_train_test_sets_ids(train_test, fold=10)
        print(":::Time used: ", time.time() - start)

        print("Running models...")
        start = time.time()
        try:
            metrics = run_model(data, current_dataset_count=count, percentage_of_top_samples=0.1)
        except:
            with open('dataset_running_order_tml_ridge_100.txt', 'a') as f:
                f.write("WARNING: Cannot build model with only one target value for Dataset " + filename + "\n")
            print("WARNING: Cannot build model with only one target value for Dataset " + filename)
            continue
        print(":::Time used: ", time.time() - start, "\n")

        all_metrics.append(metrics)
        np.save("extrapolation_10fcv_tml_ridge_100.npy", np.array(all_metrics))

