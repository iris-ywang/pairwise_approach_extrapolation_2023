"""
Test run with test_case.csv for debugging and testing the refectory
Test dataset: 20 samples, 17 features
"""

from pa_basics.import_chembl_data import dataset
from split_data import generate_train_test_sets_with_increasing_train_size
from build_model import run_model

if __name__ == '__main__':
    file = 'test_case.csv'
    train_test = dataset(file, shuffle_state=1)  # No shuffling of dataset
    # Test dataset is too small to pass the data_check(), so this step is skipped.
    data = generate_train_test_sets_with_increasing_train_size(train_test, step_size=0.8)
    # metrics = run_model(data, percentage_of_top_samples=0.2)
    print('Finished')
