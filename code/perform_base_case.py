import numpy as np
from sklearn.ensemble import RandomForestRegressor

from split_data import generate_train_test_sets
from ScoreBasedTrueSkill.score_based_bayesian_rating import ScoreBasedBayesianRating as SBBR
from ScoreBasedTrueSkill.rating import Rating as sbbr_rating


def regression(model, train_data, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model


def rating_sbbr(Y_pa, pair_ids, y_true, train_ids):
    """
    Estimate activity values from C2-type test pairs via ScoreBasedBayesRating method

    :param Y_pa: np.array of (predicted) differences in activities for test pairs
    :param pair_ids: list of tuples, each specifying samples IDs for a pair.
            * Y and pair_ids should match; their length should be the same
    :param y_true: np.array of true activity values of all samples
    :param train_ids: list of int for training sample IDs
    :return: np.array of estimated activity values for all samples (both train and test samples)
    """
    n_samples = len(y_true)
    n_comparisons = len(Y_pa)
    mean_train = np.mean(y_true[train_ids])
    dev_train = mean_train / 3
    beta = mean_train / 6
    # initialising initial rating scores (and its deviations) for all samples
    ranking = [[sbbr_rating(mean_train, dev_train, beta)] for _ in range(n_samples)]

    # updating each sample's rating score based on their (predicted) pairwise comparison
    for pair_id in range(n_comparisons):
        id_a, id_b = pair_ids[pair_id]
        comp_result = Y_pa[pair_id]
        SBBR([ranking[id_a], ranking[id_b]], [comp_result, 0]).update_skills()

    return np.array([rating_score[0].mean for rating_score in ranking])


def estimate_y_from_ranking(Y_pa_c2, Y_pa_c3, datum):
    """
    Estimate activity values from test pairs via ScoreBasedBayesRating method

    :param datum: a dict - keys = type of data, values = (pairwise) train/test samples ([y, x1, x2, ...]) OR
     (pairwise) train/test IDs
    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairs
    :param Y_pa_c3: np.array of (predicted) differences in activities for C3-type test pairs
    :return: np.array of estimated activity values for test set
    """
    c123_test_pair_ids = datum['train_pair_ids'] + \
                         datum['c2_test_pair_ids'] + \
                         datum['c3_test_pair_ids']
    Y_pa_c123 = np.concatenate((datum['train_pairs'][:, 0], Y_pa_c2, Y_pa_c3), axis=0)
    y_ranking_c123 = rating_sbbr(Y_pa_c123, c123_test_pair_ids,
                                 datum['y_true'],
                                 datum['train_ids'])
    return y_ranking_c123[datum['test_ids']]


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def generate_meta_data(all_data):
    """
    :param all_data: a dict - keys = type of data, values = (pairwise) train/test samples ([y, x1, x2, ...]) OR
     (pairwise) train/test IDs
    :return: np.array - shape = (number_test_samples, number_base_models)
    """
    # regression on FP
    _, y_SA = regression(RandomForestRegressor(), all_data['train_set'], all_data['test_set'])

    # regression on pairs of FP for C2 and C3 type test pairs
    _, Y_pa_c2 = regression(RandomForestRegressor(), all_data['train_pairs'], all_data['c2_test_pairs'])

    _, Y_pa_c3 = regression(RandomForestRegressor(), all_data['train_pairs'], all_data['c3_test_pairs'])

    # estimate activity values from C2-type test pairs via arithmetic mean
    y_EstimateFromYpa = estimate_y_from_averaging(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                                  all_data['y_true'])

    # estimate activity values from C2-type test pairs via ScoreBasedBayesRating method
    y_RankFromYpa = estimate_y_from_ranking(Y_pa_c2, Y_pa_c3, all_data)
    return np.array([y_SA, y_EstimateFromYpa, y_RankFromYpa]).T


def run_base_models(data: dict):
    """
    For each fold of outer CV, split data again to create inner CV using the training samples;
    Then for each fold of inner CV based on training set, build base models on inner-training set, and get predictions
    for inner-test set from base models;
    After inner CV, each training sample has been predicted by each base model once. Their predictions are stored as
    training set for meta-model for this outer CV.

    :param data: a dict - keys = (outer) fold number, values = the corresponding pre-processed training and test data and
             sample information
    :return: a dict - keys = (outer) fold number, values = a tuple of features and target values for meta-model
    """
    meta_data = {}
    for outer_fold, datum in data.items():
        x_meta = []
        y_meta = []
        train_set = datum['train_set']
        stack_data = generate_train_test_sets(train_set)

        for inner_fold, stack_datum in stack_data.items():
            x_meta += list(generate_meta_data(stack_datum))
            y_meta += list(stack_datum['test_set'][:, 0])

        meta_data[outer_fold] = (np.array(x_meta), np.array(y_meta))
    return meta_data
