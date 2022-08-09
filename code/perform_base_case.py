import split_data
from split_data import generate_train_test_sets
from ScoreBasedTrueSkill.score_based_bayesian_rating import ScoreBasedBayesianRating as SBBR
from ScoreBasedTrueSkill.rating import Rating as sbbr_rating

import numpy as np
from sklearn.ensemble import RandomForestRegressor


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


def rating_sbbr(comparison_results_lst, test_combs_lst, y_true, train_ids):
    nsamples = len(y_true)
    ncomparisons = len(comparison_results_lst)
    mean_train = np.mean(y_true[train_ids])
    dev_train = mean_train / 3
    beta = mean_train / 6
    ranking = [[sbbr_rating(mean_train, dev_train, beta)] for idx in range(nsamples)]

    for comp_id in range(ncomparisons):
        ida, idb = test_combs_lst[comp_id]
        comp_result = comparison_results_lst[comp_id]
        SBBR([ranking[ida], ranking[idb]], [comp_result, 0]).update_skills()

    return np.array([i[0].mean for i in ranking])


def weighted_estimate(y_pw_lst, test_combs_lst, test_ids, y_true, Y_prob=None):
    if y_true is None:
        y_true = y_true
    if Y_prob is None:  # linear arithmetic
        Y_prob = np.ones((len(y_pw_lst)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(y_pw_lst)):
        ida, idb = test_combs_lst[pair]
        delta_ab = y_pw_lst[pair]
        weight = Y_prob[pair]

        if ida in test_ids:
            # (stest, strain)
            w_esti = (y_true[idb] + delta_ab) * weight
            records[ida] += w_esti
            weights[ida] += weight

        elif idb in test_ids:
            # (strain, stest)
            w_esti = (y_true[ida] - delta_ab) * weight
            records[idb] += w_esti
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


def estimate_y_from_ranking(Y_pa_c2, Y_pa_c3, stack_datum):
    c123_test_pair_ids = stack_datum['train_pair_ids'] + \
                         stack_datum['c2_test_pair_ids'] + \
                         stack_datum['c3_test_pair_ids']
    Y_pa_c123 = np.concatenate((stack_datum['train_pairs'][:,0], Y_pa_c2, Y_pa_c3), axis=0)
    y_ranking_c123 = rating_sbbr(Y_pa_c123, c123_test_pair_ids,
                                stack_datum['y_true'],
                                stack_datum['train_ids'])
    return y_ranking_c123[stack_datum['test_ids']]


def generate_meta_data(all_data):
    # regression on FP
    _, y_sa = regression(RandomForestRegressor(), all_data['train_set'], all_data['test_set'])

    # regression on pairs of FP
    _, Y_pa_c2 = regression(RandomForestRegressor(), all_data['train_pairs'], all_data['c2_test_pairs'])

    _, Y_pa_c3 = regression(RandomForestRegressor(), all_data['train_pairs'], all_data['c3_test_pairs'])

    #
    y_EstimateFromYpa = weighted_estimate(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                          all_data['y_true'])

    y_RankFromYpa = estimate_y_from_ranking(Y_pa_c2, Y_pa_c3, all_data)
    return np.array([y_sa, y_EstimateFromYpa, y_RankFromYpa]).T


def run_base_models(data: dict):
    
    meta_data = {}

    for outer_fold, fold_datum in data.items():
        x_meta = []
        y_meta = []
        train_set = fold_datum['train_set']
        stack_data = generate_train_test_sets(train_set)
        
        for inner_fold, stack_datum in stack_data.items():
            x_meta += list(generate_meta_data(stack_datum))
            y_meta += list(stack_datum['test_set'][:, 0])
        
        meta_data[outer_fold] = (np.array(x_meta), np.array(y_meta))
    return meta_data
