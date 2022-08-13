import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize

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


def run_stacking(data: dict, meta_data: dict) -> np.ndarray:
    """
    For each fold:
    Take the base-model predictions from trainings samples and their true values, build the meta model;
    Then re-build the base-models using all of the training samples, predict for test samples, which are then input
    to the meta-model to get a final predictions for test samples.
    :param data: a dict - keys = (outer) fold number, values = the corresponding pre-processed training and test data and
             sample information
    :param meta_data: a dict - keys = (outer) fold number, values = a tuple of features and target values for meta-model
    :return: np.array of metrics, shape = (number_fold, number_of_base+1, number_of_metric)
    """
    metrics = []
    for outer_fold, meta_datum in meta_data.items():
        x_meta_train, y_meta_train = meta_datum
        ms = constrained_linear_regression()
        meta_model = ms.fit(x_meta_train, y_meta_train)

        # generate x_meta_test
        predictions_base = generate_meta_data(data[outer_fold])
        x_meta_test = np.array(predictions_base)
        y_meta_test = data[outer_fold]['test_set'][:, 0]
        y_prediction_meta = meta_model.predict(x_meta_test)
        metrics_per_fold = meta_evaluation(predictions_base, y_prediction_meta, y_meta_test)
        metrics.append(metrics_per_fold)
    return np.array(metrics)

