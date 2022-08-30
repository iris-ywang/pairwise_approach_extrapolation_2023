import os

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets
from perform_base_case import run_base_models
from perform_stacking import run_stacking


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
    all_metrics = []

    for file in range(len(chembl_info)):
        if len(all_metrics) > 150: break
        if chembl_info["Repetition Rate"][file] > 0.15: continue
        if chembl_info["N(sample)"][file] > 300 or chembl_info["N(sample)"][file] < 90: continue
        # If dataset passes the above criteria, then it gives a dict of fold number and their corresponding
        # pre-processed data

        # TODO: may need to change the way of getting parent directory if this does not work on windows
        connection = "/input/qsar_data_unsorted/"
        train_test = dataset(os.getcwd() + connection + chembl_info["File name"][file])
        data = generate_train_test_sets(train_test)

        meta_data = run_base_models(data)  # a dict: key = fold number, values = (x_meta, y_meta)
        metrics = run_stacking(data,
                               meta_data)  # np array: shape = (number_of_fold, number_of_base+1, number_of_metric)
        all_metrics.append(metrics)

