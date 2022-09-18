import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score,\
    accuracy_score, confusion_matrix, f1_score, precision_score
from scipy.stats import spearmanr, kendalltau


def build_ml_model(model, train_data, test_data=None):
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


def classification_evaluation(y_true, y_pred):
    acc = accuracy_score(np.sign(y_true), np.sign(y_pred))
    prec = precision_score(np.sign(y_true), np.sign(y_pred), average="macro", zero_division=0)
    cm = f1_score(np.sign(y_true), np.sign(y_pred), average='macro', zero_division=0)
    return [acc, prec, cm]


def distance_evaluation(y_true, y_pred):
    mse = mean_squared_error(abs(y_true), abs(y_pred))
    mae = mean_absolute_error(abs(y_true), abs(y_pred))
    r2 = r2_score(abs(y_true), abs(y_pred))
    return [mse, mae, r2]


def pairwise_differences_for_standard_approach(all_data, y_pred_all, type_of_test_pair_ids):
    Y_derived = []
    for pair_id in all_data[type_of_test_pair_ids]:
        id_a, id_b = pair_id
        Y_derived.append(y_pred_all[id_a] - y_pred_all[id_b])
    return np.array(Y_derived)


def performance_standard_approach(all_data):
    _, y_SA = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_set'],
                             all_data['test_set'])
    y_pred_all = np.array(all_data["y_true"])
    y_pred_all[all_data["test_ids"]] = y_SA

    Y_c2_derived = pairwise_differences_for_standard_approach(all_data, y_pred_all, 'c2_test_pair_ids')
    Y_c3_derived = pairwise_differences_for_standard_approach(all_data, y_pred_all, 'c3_test_pair_ids')
    return [classification_evaluation(np.sign(all_data["c2_test_pairs"][:, 0]), np.sign(Y_c2_derived)),
            distance_evaluation(abs(all_data["c2_test_pairs"][:, 0]), abs(Y_c2_derived)),
            classification_evaluation(np.sign(all_data["c3_test_pairs"][:, 0]), np.sign(Y_c3_derived)),
            distance_evaluation(abs(all_data["c3_test_pairs"][:, 0]), abs(Y_c3_derived))]


def performance_pairwise_approach_in_sign_and_abs(all_data, type_of_test_pairs: str):
    train_pairs_for_sign = np.array(all_data["train_pairs"])
    train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
    test_pairs_for_sign = np.array(all_data[type_of_test_pairs])
    test_pairs_for_sign[:, 0] = np.sign(test_pairs_for_sign[:, 0])
    _, Y_pa_sign = build_ml_model(RandomForestClassifier(n_jobs=-1, random_state=1), train_pairs_for_sign,
                                  test_pairs_for_sign)

    train_pairs_for_abs = np.absolute(all_data["train_pairs"])
    test_pairs_for_abs = np.absolute(all_data[type_of_test_pairs])
    _, Y_pa_abs = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1), train_pairs_for_abs,
                                 test_pairs_for_abs)

    return [classification_evaluation(test_pairs_for_sign[:, 0], Y_pa_sign), \
            distance_evaluation(test_pairs_for_abs[:, 0], Y_pa_abs)]


def performance_pairwise_approach(all_data):
    # regression on pairs of FP for C2 and C3 type test pairs
    _, Y_pa_c2 = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1),
                                all_data['train_pairs'], all_data['c2_test_pairs'])
    _, Y_pa_c3 = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1),
                                all_data['train_pairs'], all_data['c3_test_pairs'])

    # estimate activity values from C2-type test pairs via arithmetic mean
    y_EstimateFromYpa = estimate_y_from_averaging(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                                  all_data['y_true'])

    metrics_Yc2_together = [classification_evaluation(np.sign(all_data['c2_test_pairs'][:, 0]), np.sign(Y_pa_c2)),
                            distance_evaluation(abs(all_data['c2_test_pairs'][:, 0]), abs(Y_pa_c2))]
    metrics_Yc3_together = [classification_evaluation(np.sign(all_data['c3_test_pairs'][:, 0]), np.sign(Y_pa_c3)),
                            distance_evaluation(abs(all_data['c3_test_pairs'][:, 0]), abs(Y_pa_c3))]
    metrics_Yc2_separate = performance_pairwise_approach_in_sign_and_abs(all_data, "c2_test_pairs")
    metrics_Yc3_separate = performance_pairwise_approach_in_sign_and_abs(all_data, "c3_test_pairs")

    return metrics_Yc2_together + metrics_Yc3_together + metrics_Yc2_separate + metrics_Yc3_separate


def run_model(data):
    metrics = []
    for outer_fold, datum in data.items():
        metric_sa = performance_standard_approach(datum) # a list of 2 * 2 metrics
        metric_pa = performance_pairwise_approach(datum) # a list of 2 * 4 metrics
        metrics.append(metric_sa + metric_pa)
    return metrics
