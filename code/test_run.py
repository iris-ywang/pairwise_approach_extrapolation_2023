"""
Test run with test_case.csv for debugging and testing the refectory
Test dataset: 20 samples, 17 features
"""

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_with_K_fold_forward
from build_model import run_model

if __name__ == '__main__':
    file = 'test_case.csv'
    train_test = dataset(file, shuffle_state=1)  # No shuffling of dataset
    sorted_train_test = train_test[train_test[:, 0].argsort()]
    # Test dataset is too small to pass the data_check(), so this step is skipped.
    metrics = generate_train_test_sets_with_K_fold_forward(sorted_train_test, folds=4)
    print('Finished')
