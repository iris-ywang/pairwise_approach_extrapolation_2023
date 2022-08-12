from split_data import generate_train_test_sets
from pa_basics.import_chembl_data import dataset
from perform_base_case import run_base_models
from perform_stacking import run_stacking

file = 'test_case.csv'
train_test = dataset(file)
data = generate_train_test_sets(train_test)

meta_data = run_base_models(data)
metrics = run_stacking(data, meta_data)