
from pa_basics.all_pairs import paired_data
from pa_basics.import_chembl_data import dataset
import numpy as np
from sklearn.model_selection import KFold
from itertools import permutations, product


def data_check(train_test):
    sample_size = np.shape(train_test)[0]
    if sample_size > 100 or sample_size < 90: return False

    my_dict = {i: list(train_test[:, 0]).count(i) for i in list(train_test[:, 0])}
    max_repetition = max(my_dict.values())
    if max_repetition > 0.15 * sample_size: return False

    return True


def train_test_ids(train_test, cv, fold):
    sample_size = np.shape(train_test)[0]
    ntest = int(sample_size / fold)
    train_ids = list(range(sample_size))
    if cv == fold - 1:
        start = sample_size - (sample_size - ntest * cv)
        test_ids = list(range(start, sample_size))
        del train_ids[start: sample_size]
    else:
        test_ids = list(range(cv * ntest, (cv + 1) * ntest))
        del train_ids[(cv * ntest): ((cv + 1) * ntest)]
    return train_ids, test_ids


def train_test_split(pairs, train_ids, test_ids):
    def pair_test_w_train(train_ids, test_ids):
        c2test_combs = []
        for comb in product(test_ids, train_ids):
            c2test_combs.append(comb)
            c2test_combs.append(comb[::-1])
        return c2test_combs

    train_pairs = dict(pairs)
    c2_test_pairs = []
    c3_test_pairs = []
    c2_keys_del = pair_test_w_train(train_ids, test_ids)
    c3_keys_del = list(permutations(test_ids, 2)) + [(a, a) for a in test_ids]

    for key in c2_keys_del:
        c2_test_pairs.append(train_pairs.pop(key))
    for key in c3_keys_del:
        c3_test_pairs.append(train_pairs.pop(key))
    c2_test_pairs = np.array(c2_test_pairs)
    c3_test_pairs = np.array(c3_test_pairs)

    c1_keys_del, trainp = [], []
    for a, b in train_pairs.items():
        c1_keys_del.append(a)
        trainp.append(b)
    train_pairs = np.array(trainp)

    return {'train_pairs': train_pairs,
            'train_pair_ids': c1_keys_del,
            'c2_test_pairs': c2_test_pairs,
            'c2_test_pair_ids': c2_keys_del,
            'c3_test_pairs': c3_test_pairs,
            'c3_test_pair_ids': c3_keys_del}


def generate_train_test_sets(train_test):
    fold = 5
    y_true = np.array(train_test[:, 0])
    pairs = paired_data(train_test)

    train_test_data = {}
    kf = KFold(n_splits=fold)
    n_fold = 0
    for train_ids, test_ids in kf.split(train_test):
        train_test_data_per_fold = {}
        train_test_data_per_fold['train_ids'] = train_ids
        train_test_data_per_fold['test_ids'] = test_ids

        train_test_data_per_fold['train_set'] = train_test[train_ids]
        train_test_data_per_fold['test_set'] = train_test[test_ids]
        train_test_data_per_fold['y_true'] = y_true

        pairs_data = train_test_split(pairs, train_ids, test_ids)
        train_test_data[n_fold] = {**train_test_data_per_fold, **pairs_data}
        
        n_fold += 1

    return train_test_data


def load_and_check_data(filename):
    train_test = dataset(filename)
    if data_check(train_test):
        data = generate_train_test_sets(train_test)
        return data
    else:
        return None
