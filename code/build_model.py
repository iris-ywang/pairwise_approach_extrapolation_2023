import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from extrapolation_evaluation import EvaluateAbilityToIdentifyTopTestSamples
from pa_basics.all_pairs import paired_data_by_pair_id
from pa_basics.rating import rating_trueskill


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


def performance_standard_approach(all_data, percentage_of_top_samples):
    sa_model, y_SA = build_ml_model(RandomForestRegressor(n_jobs=-1, random_state=1), all_data['train_set'], all_data['test_set'])
    y_pred_all = np.array(all_data["y_true"])
    y_pred_all[all_data["test_ids"]] = y_SA

    metrics = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_pred_all, all_data).run_evaluation()
    return metrics, sa_model


def performance_pairwise_approach(all_data, percentage_of_top_samples, batch_size=200000):
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size

    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], all_data['train_pair_ids'])

        train_pairs_for_sign = np.array(train_pairs_batch)
        train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
        rfc = RandomForestClassifier(n_jobs=-1, random_state=1)
        rfc = build_ml_model(rfc, train_pairs_for_sign)

        train_pairs_for_abs = np.absolute(train_pairs_batch)
        rfr = RandomForestRegressor(n_jobs=-1, random_state=1)
        rfr = build_ml_model(rfr, train_pairs_for_abs)

    else:

        for run in range(runs_of_estimators + 1):
            if run < runs_of_estimators:
                train_ids_per_batch = all_data["train_pair_ids"][run*batch_size:(run + 1) * batch_size]

            else:
                train_ids_per_batch = all_data["train_pair_ids"][run*batch_size:]

            train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], train_ids_per_batch)

            train_pairs_for_sign = np.array(train_pairs_batch)
            train_pairs_for_sign[:, 0] = np.sign(train_pairs_for_sign[:, 0])
            rfc = RandomForestClassifier(n_jobs=-1, random_state=1, warm_start=True)
            rfc = build_ml_model(rfc, train_pairs_for_sign)

            train_pairs_for_abs = np.absolute(train_pairs_batch)
            rfr = RandomForestRegressor(n_jobs=-1, random_state=1, warm_start=True)
            rfr = build_ml_model(rfr, train_pairs_for_abs)

            rfc.n_estimators += 100
            rfr.n_estimators += 100

    c2_test_pair_ids = all_data["c2_test_pair_ids"]
    number_test_batches = len(c2_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c2_sign, Y_pa_c2_dist = [], []
    Y_pa_c2_true = []
    for test_batch in range(number_test_batches + 1):
        test_pair_id_batch = c2_test_pair_ids[
                             test_batch * batch_size: (test_batch + 1) * batch_size]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        print("Shape of test_pairs_batch: ", np.shape(test_pairs_batch))
        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_dist += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c2_true += list(test_pairs_batch[:, 0])
        if (test_batch + 1) * batch_size >= len(c2_test_pair_ids): break


    Y_c2_sign_and_abs_predictions = dict(zip(all_data["c2_test_pair_ids"], np.array([Y_pa_c2_dist, Y_pa_c2_sign]).T))
    y_ranking = rating_trueskill(Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"])

    metrics = EvaluateAbilityToIdentifyTopTestSamples(percentage_of_top_samples, all_data["y_true"],
                                                      y_ranking, all_data).run_evaluation(Y_c2_sign_and_abs_predictions)
    return metrics, rfc, rfr


def run_model(data, percentage_of_top_samples):
    metrics = []
    for outer_fold, datum in data.items():

        metric_sa, rfr_sa = performance_standard_approach(datum, percentage_of_top_samples)
        metric_pa, rfc_pa, rfr_pa = performance_pairwise_approach(datum, percentage_of_top_samples)
        metrics.append([metric_sa, metric_pa])

    return np.array([metrics])
