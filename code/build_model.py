import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from pa_basics.rating import rating_trueskill


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


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    ndcg = ndcg_score([y_true], [y_predict])
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    tau = kendalltau(y_true, y_predict)[0]
    r2 = r2_score(y_true, y_predict)
    return [mse, mae, r2, rho, ndcg, tau]


def performance_standard_approach(datum):
    _, y_SA = regression(RandomForestRegressor(n_jobs=-1, random_state=1), datum['train_set'], datum['test_set'])
    return metrics_evaluation(datum["test_set"][:, 0], y_SA)


def pairwise_differences_for_standard_approach(all_data, y_pred_all):
    Y_c2_abs_derived = []
    for pair_id in all_data["c2_test_pair_ids"]:
        id_a, id_b = pair_id
        Y_c2_abs_derived.append(y_pred_all[id_a] - y_pred_all[id_b])
    return np.array(Y_c2_abs_derived)


def estimate_averaged_y_from_final_ranking(all_data, y_pred_all, Y_c2_sign_and_abs_predictions):
    overall_orders = np.argsort(-y_pred_all)  # a list of sample IDs in the descending order of activity values
    y_test_estimations = []
    for test_id in all_data["test_ids"]:
        test_order_position = int(np.where(overall_orders == test_id)[0])
        test_estimates = []
        for train_id in all_data['train_ids']:
            train_order_position = int(np.where(overall_orders == train_id)[0])
            if test_order_position < train_order_position:
                # i.e.test is ranked higher than the top train -> higher activity than that of the top train sample
                test_estimates.append(Y_c2_sign_and_abs_predictions[(test_id, train_id)][0] * 1
                                      + all_data["y_true"][train_id])
                # No need of the below because:
                #     sv[(test_id, train_id)] = sv[(train_id, test_id)] = abs(y_train_id - y_test_id)
                # test_estimates.append(sv[(train_id, test_id)][0] * -1 + self.y_true[train_id])
            elif test_order_position > train_order_position:
                test_estimates.append(Y_c2_sign_and_abs_predictions[(test_id, train_id)][0] * -1
                                      + all_data["y_true"][train_id])
        y_test_estimations.append(np.mean(test_estimates))
    return y_test_estimations

def performance_pairwise_approach(all_data):
    # regression on pairs of FP for C2 and C3 type test pairs
    # _, Y_pa_c2 = regression(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_pairs'], all_data['c2_test_pairs'])
    # _, Y_pa_c3 = regression(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_pairs'], all_data['c3_test_pairs'])

    train_pairs_for_sign = np.array(all_data["train_pairs"])
    train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
    _, Y_pa_c2_sign = regression(RandomForestClassifier(n_jobs=-1, random_state=1), train_pairs_for_sign,
                                     all_data['c2_test_pairs'])

    train_pairs_for_abs = np.absolute(all_data["train_pairs"])
    c2_test_pairs_for_abs = np.absolute(all_data['c2_test_pairs'])
    _, Y_pa_c2_abs = regression(RandomForestRegressor(n_jobs=-1, random_state=1), train_pairs_for_abs,
                                    c2_test_pairs_for_abs)
    # Y_c2_sign_and_abs_predictions = dict(zip(all_data["c2_test_pair_ids"], np.array([Y_pa_c2_abs, Y_pa_c2_sign]).T))
    y_ranking = rating_trueskill(Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"])
    Y_pa_c2 = np.sign(pairwise_differences_for_standard_approach(all_data, y_ranking)) * Y_pa_c2_abs
    y_EstimateFromRankNDistance = estimate_y_from_averaging(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                                            all_data['y_true'])
    # y_EstimateFromRankNDistance = estimate_averaged_y_from_final_ranking(all_data, y_ranking,
    #                                                                       Y_c2_sign_and_abs_predictions)
    # metrics_Yc2 = metrics_evaluation(all_data["c2_test_pairs"][:, 0], Y_pa_c2)
    # metrics_Yc3 = metrics_evaluation(all_data["c3_test_pairs"][:, 0], Y_pa_c3)
    metrics_y_est = metrics_evaluation(all_data["test_set"][:, 0], y_EstimateFromRankNDistance)
    # print(spearmanr(all_data['y_true'][all_data['test_ids']], y_ranking[all_data['test_ids']]),
    #       spearmanr(all_data['y_true'][all_data['test_ids']], y_EstimateFromRankNDistance))
    return metrics_y_est


def run_model(data):
    metrics = []
    for outer_fold, datum in data.items():
        metric_sa = performance_standard_approach(datum)
        metric_pa = performance_pairwise_approach(datum)
        metrics.append([metric_sa] + [metric_pa])
    return metrics
