"""
Test run with test_case.csv for debugging and testing the refectory
Test dataset: 20 samples, 17 features
"""

from split_data import generate_train_test_sets
from pa_basics.import_chembl_data import dataset
from perform_base_case import run_base_models
from perform_stacking import run_stacking

file = 'test_case.csv'
train_test = dataset(file, shuffle_state=None)  # No shuffling of dataset
# Test dataset is too small to pass the data_check(), so this step is skipped.
data = generate_train_test_sets(train_test)

meta_data = run_base_models(data)
metrics = run_stacking(data, meta_data)
