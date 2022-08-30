"""
Test run with test_case.csv for debugging and testing the refectory
Test dataset: 20 samples, 17 features
"""

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets
from build_model import run_model

if __name__ == '__main__':
    file = 'test_case.csv'
    train_test = dataset(file, shuffle_state=None)  # No shuffling of dataset
    # Test dataset is too small to pass the data_check(), so this step is skipped.
    data = generate_train_test_sets(train_test, only_fp=True)
    metrics = run_model(data)
    print('Finished')
