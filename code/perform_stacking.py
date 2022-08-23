import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from perform_base_case import generate_meta_data


class constrained_linear_regression():
    """
    Create a linear regression model whose coefficients are positive and their sum equals to 1. The class is formulated to work in a similar
    way with scikit-learn LinearRegressor.
    """

    def __init__(self, positive=True, sum_one=True, coef=None):
        self.bnds = positive
        self.cons = sum_one
        self.coef_ = coef

    def fit(self, X, Y):
        nsamples, nfeatures = np.shape(X)
        # Define the Model
        model = lambda b, X: sum([b[i] * X[:, i] for i in range(nfeatures)])

        # The objective Function to minimize (least-squares regression)
        obj = lambda b, Y, X: np.sum(np.abs(Y - model(b, X)) ** 2)

        # Bounds: b[0], b[1], b[2] >= 0

        bnds = [(0, None) for _ in range(nfeatures)] if self.bnds else None

        # Constraint: b[0] + b[1] + b[2] - 1 = 0
        cons = [{"type": "eq", "fun": lambda b: sum(b) - 1}] if self.bnds else ()

        # Initial guess for b[1], b[2], b[3]:
        xinit = np.array([1] + [0 for _ in range(nfeatures - 1)])

        res = minimize(obj, args=(Y, X), x0=xinit, bounds=bnds, constraints=cons)

        return constrained_linear_regression(self.bnds, self.cons, res.x)

    def predict(self, Xtest):
        return np.matmul(Xtest, self.coef_)


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    ndcg = ndcg_score([y_true], [y_predict])
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    tau = kendalltau(y_true, y_predict)[0]
    r2 = r2_score(y_true, y_predict)
    return mse, mae, r2, rho, ndcg, tau


def meta_evaluation(predictions_base, prediction_meta, y_true_test):
    """
    Evaluate the model accuracy for both base-models and meta-model.

    :param predictions_base: np.array - shape = (number_test_samples, number_base_models)
    :param prediction_meta: np.array - shape = (number_test_samples,)
    :param y_true_test: np.array of true activity values of test samples
    :return: np.array, shape = (number_base_model + 1, number_metric)
    """
    n_test, n_base = np.shape(predictions_base)
    metrics = np.empty((n_base + 1, 6))
    metrics[0, :] = metrics_evaluation(y_true_test, prediction_meta)

    for i in range(n_base):
        metrics[i + 1, :] = metrics_evaluation(y_true_test, predictions_base[:, i])
    return metrics


def transform_meta_class_forward(x_meta_train, y_meta_train):
    """
    base    p1        p2        actual        class

0.5        0.6        0.7        0.4        1

0.5        0.55      0.7        0.6        2

0.5        0.45      0.6        0.4        3

    Note: the first (0th) column(feature) of x_meta_train must be from the standard approach
    :param x_meta_train:
    :param y_meta_train:
    :return:
    """
    y_meta_class = np.empty(np.shape(y_meta_train))
    y_true = np.array([y_meta_train]).T
    diff = x_meta_train - y_true
    diff_abs = np.absolute(diff)
    min_column_id = np.argmin(diff_abs, axis=1)  # find the predictions closest to y_true
    best_values = x_meta_train[np.arange(len(x_meta_train)), min_column_id]
    for sample in range(len(y_meta_class)):
        if min_column_id[sample] == 0:
            y_meta_class[sample] = 1
            continue

        if best_values[sample] > x_meta_train[sample, 0]:
            y_meta_class[sample] = 2
        elif best_values[sample] < x_meta_train[sample, 0]:
            y_meta_class[sample] = 3
    return y_meta_class


def transform_meta_class_backward(x_meta_train, y_meta_class):
    y_meta_value = np.empty(np.shape(y_meta_class))
    for sample in range(len(y_meta_class)):
        prediction_sa = x_meta_train[sample, 0]
        if y_meta_class[sample] == 1:
            y_meta_value[sample] = prediction_sa
        elif y_meta_class[sample] == 2:
            sample_predictions = x_meta_train[sample]
            try:
                y_meta_value[sample] = sample_predictions[sample_predictions > prediction_sa].min()
            except ValueError:
                y_meta_value[sample] = prediction_sa
        elif y_meta_class[sample] == 3:
            sample_predictions = x_meta_train[sample]
            try:
                y_meta_value[sample] = sample_predictions[sample_predictions < prediction_sa].max()
            except ValueError:
                y_meta_value[sample] = prediction_sa
    return y_meta_value


def run_stacking(data: dict, meta_data: dict) -> np.ndarray:
    """
    # For each fold:
    # Take the base-model predictions from trainings samples and their true values, build the meta model;
    # Then re-build the base-models using all of the training samples, predict for test samples, which are then input
    # to the meta-model to get a final predictions for test samples.
    # :param data: a dict - keys = (outer) fold number, values = the corresponding pre-processed training and test data and
    #          sample information
    # :param meta_data: a dict - keys = (outer) fold number, values = a tuple of features and target values for meta-model
    # :return: np.array of metrics, shape = (number_fold, number_of_base+1, number_of_metric)
    """
    metrics = []
    for outer_fold, meta_datum in meta_data.items():
        x_meta_train, y_meta_train = meta_datum
        y_meta_class = transform_meta_class_forward(x_meta_train, y_meta_train)
        unique, counts = np.unique(y_meta_class, return_counts=True)
        print(np.asarray((unique, counts)).T)
        ms = LogisticRegression(n_jobs=-1)
        meta_model = ms.fit(x_meta_train, y_meta_class)

        # generate x_meta_test
        predictions_base = generate_meta_data(data[outer_fold])
        x_meta_test = np.array(predictions_base)
        y_meta_test = data[outer_fold]['test_set'][:, 0]
        y_class_meta = meta_model.predict(x_meta_test)
        y_prediction_meta = transform_meta_class_backward(x_meta_test, y_class_meta)

        metrics_per_fold = meta_evaluation(predictions_base, y_prediction_meta, y_meta_test)
        metrics.append(metrics_per_fold)
    return np.array(metrics)


"""
Question:
base    p1        p2        actual        class

0.5        0.6        0.7        0.8        2

But when it is converted back, it would choose 0.6 but not 0.7
"""
