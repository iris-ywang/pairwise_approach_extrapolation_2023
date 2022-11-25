import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score,\
    accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.utils import resample
from scipy.stats import spearmanr, kendalltau
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
    Y_c2_true = pairwise_differences_for_standard_approach(all_data, all_data["y_true"], 'c2_test_pair_ids')
    Y_c3_true = pairwise_differences_for_standard_approach(all_data, all_data["y_true"], 'c3_test_pair_ids')

    return [metrics_evaluation(all_data['test_set'][:, 0], y_SA)] + \
        [classification_evaluation(np.sign(Y_c2_true), np.sign(Y_c2_derived)),
            distance_evaluation(abs(Y_c2_true), abs(Y_c2_derived)),
            classification_evaluation(np.sign(Y_c3_true), np.sign(Y_c3_derived)),
            distance_evaluation(abs(Y_c3_true), abs(Y_c3_derived))]


def performance_pairwise_approach_in_sign_and_abs(all_data, batch_size=500000):
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
                train_ids_per_batch = all_data["train_pair_ids"][run*batch_size : (run + 1) * batch_size]

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
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c2_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c2_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c2_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_dist += list(rfr.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c2_true += list(test_pairs_batch[:,0])

    c3_test_pair_ids = all_data["c2_test_pair_ids"]
    number_test_batches = len(c3_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c3_sign, Y_pa_c3_dist = [], []
    Y_pa_c3_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c3_test_pair_ids[
                                 test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c3_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c3_sign += list(rfc.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_dist += list(rfc.predict(np.absolute(test_pairs_batch[:, 1:])))
        Y_pa_c3_true += list(test_pairs_batch[:, 0])

    y_ranking = rating_trueskill(Y_pa_c2_sign, all_data["c2_test_pair_ids"], all_data["y_true"])
    Y_pa_c2 = np.sign(pairwise_differences_for_standard_approach(all_data, y_ranking, "c2_test_pair_ids")) * Y_pa_c2_dist
    y_EstimateFromRankNDistance = estimate_y_from_averaging(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                                            all_data['y_true'])
    metrics_for_y = metrics_evaluation(all_data['y_true'][all_data['test_ids']], y_EstimateFromRankNDistance)

    metrics_Yc2_together = [classification_evaluation(np.sign(Y_pa_c2_true), np.sign(Y_pa_c2_sign)),
                            distance_evaluation(abs(np.array(Y_pa_c2_true)), abs(np.array(Y_pa_c2_dist)))]
    metrics_Yc3_together = [classification_evaluation(np.sign(Y_pa_c3_true), np.sign(Y_pa_c3_sign)),
                            distance_evaluation(abs(np.array(Y_pa_c3_true)), abs(np.array(Y_pa_c3_dist)))]

    return [metrics_for_y] + metrics_Yc2_together + metrics_Yc3_together


def performance_pairwise_approach(all_data, batch_size=500000):
    runs_of_estimators = len(all_data["train_pair_ids"]) // batch_size
    if runs_of_estimators < 1:
        train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], all_data['train_pair_ids'])
        rfr = RandomForestRegressor(n_jobs=-1, random_state=1)
        rfr = build_ml_model(rfr, train_pairs_batch)

    else:
        for run in range(runs_of_estimators+1):
            if run < runs_of_estimators:
                train_ids_per_batch = all_data["train_pair_ids"][run * batch_size: (run + 1) * batch_size]
            else:
                train_ids_per_batch = all_data["train_pair_ids"][run * batch_size:]

            train_pairs_batch = paired_data_by_pair_id(all_data["train_test"], train_ids_per_batch)
            rfr = RandomForestRegressor(n_jobs=-1, random_state=1, warm_start=True)
            rfr = build_ml_model(rfr, train_pairs_batch)

            rfr.n_estimators += 100


    c2_test_pair_ids = all_data["c2_test_pair_ids"]
    number_test_batches = len(c2_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c2 = []
    Y_pa_c2_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c2_test_pair_ids[
            test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c2_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c2 += list(rfr.predict(test_pairs_batch[:, 1:]))
        Y_pa_c2_true += list(test_pairs_batch[:,0])


    c3_test_pair_ids = all_data["c3_test_pair_ids"]
    number_test_batches = len(c3_test_pair_ids) // batch_size
    if number_test_batches < 1: number_test_batches = 0
    Y_pa_c3 = []
    Y_pa_c3_true = []
    for test_batch in range(number_test_batches + 1):
        if test_batch != number_test_batches + 1:
            test_pair_id_batch = c3_test_pair_ids[
                test_batch * batch_size: (test_batch + 1) * batch_size]
        else:
            test_pair_id_batch = c3_test_pair_ids[test_batch * batch_size:]
        test_pairs_batch = paired_data_by_pair_id(all_data["train_test"], test_pair_id_batch)
        Y_pa_c3 += list(rfr.predict(test_pairs_batch[:, 1:]))
        Y_pa_c3_true += list(test_pairs_batch[:, 0])


    # estimate activity values from C2-type test pairs via arithmetic mean
    y_EstimateFromYpa = estimate_y_from_averaging(Y_pa_c2, all_data['c2_test_pair_ids'], all_data['test_ids'],
                                                  all_data['y_true'])
    metrics_for_y = metrics_evaluation(all_data['y_true'][all_data['test_ids']], y_EstimateFromYpa)
    metrics_Yc2_together = [classification_evaluation(np.sign(Y_pa_c2_true), np.sign(Y_pa_c2)),
                            distance_evaluation(abs(np.array(Y_pa_c2_true)), abs(np.array(Y_pa_c2)))]
    metrics_Yc3_together = [classification_evaluation(np.sign(Y_pa_c3_true), np.sign(Y_pa_c3)),
                            distance_evaluation(abs(np.array(Y_pa_c3_true)), abs(np.array(Y_pa_c3)))]

    return [metrics_for_y] + metrics_Yc2_together + metrics_Yc3_together

def flatten(l):
    return [item for sublist in l for item in sublist]

def run_model(data, current_dataset_count):
    temporary_file_dataset_count = int(np.load("temporary_dataset_count.npy"))

    if current_dataset_count == temporary_file_dataset_count:
        existing_iterations = np.load("PA_all_data_temporary.npy")
        existing_count = len(existing_iterations)
        metrics = list(existing_iterations)
    else:
        metrics = []
        existing_count = 0

    count = 0
    for outer_fold, datum in data.items():
        count += 1
        if count <= existing_count: continue

        metric_sa = flatten(performance_standard_approach(datum))
        metric_pa = flatten(performance_pairwise_approach(datum))
        metric_pa_signNdist = flatten(performance_pairwise_approach_in_sign_and_abs(datum))
        metrics.append([metric_sa, metric_pa, metric_pa_signNdist])


        np.save("temporary_dataset_count.npy", [current_dataset_count])
        np.save("PA_all_data_temporary.npy", np.array(metrics))

    return np.array([metrics])
