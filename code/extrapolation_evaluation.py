import numpy as np
from sklearn.metrics import mean_squared_error


class EvaluateAbilityToIdentifyTopTestSamples:
    def __init__(self, percentage_of_top_samples, y_train_with_true_test, y_train_with_predicted_test, train_ids,
                 test_ids):
        self.y_pred_all = y_train_with_predicted_test
        self.y_true_all = y_train_with_true_test
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.pc = percentage_of_top_samples

    def find_top_test_ids(self, y_pred_all, Y_sign_and_abs_predictions=None):
        # trains == train samples; tests == test samples.

        overall_orders = np.argsort(-y_pred_all)  # a list of sample IDs in the descending order of activity values
        top_trains_and_tests = overall_orders[0: int(self.pc * len(overall_orders))]
        top_tests = [idx for idx in top_trains_and_tests if idx in self.test_ids]

        # Find the ID of top train sample in the overall_order
        top_train_order_position = 0
        while True:
            if overall_orders[top_train_order_position] in self.train_ids: break
            top_train_order_position += 1
        top_train_id = overall_orders[top_train_order_position]

        tests_better_than_top_train = list(overall_orders[:top_train_order_position])

        if Y_sign_and_abs_predictions is None:
            return top_tests, tests_better_than_top_train, top_train_id
        else:
            final_estimate_of_y_and_delta_y = self.estimate_y_from_final_ranking_and_absolute_Y(
                top_tests, tests_better_than_top_train, top_train_id, overall_orders, Y_sign_and_abs_predictions
            )
            return top_tests, tests_better_than_top_train, top_train_id, final_estimate_of_y_and_delta_y

    def estimate_y_from_final_ranking_and_absolute_Y(self, top_tests, tests_better_than_top_train, top_train_id,
                                                     overall_orders, Y_sign_and_abs_predictions):
        final_estimate_of_y_and_delta_y = {}
        for test_id in (top_tests + tests_better_than_top_train):
            test_estimate = self.estimate_averaged_y_from_final_ranking(overall_orders, test_id,
                                                                        Y_sign_and_abs_predictions)
            if test_id in tests_better_than_top_train:
                Y_rough = Y_sign_and_abs_predictions[(test_id, top_train_id)][0] * 1
            if test_id in top_tests:
                Y_rough = Y_sign_and_abs_predictions[(test_id, top_train_id)][0] * -1

            Y_ave = test_estimate - self.y_true_all[top_train_id]
            final_estimate_of_y_and_delta_y[test_id] = [Y_rough, Y_ave, self.y_true_all[test_id], test_estimate]
        return final_estimate_of_y_and_delta_y

    def estimate_averaged_y_from_final_ranking(self, overall_orders, test_id, Y_sign_and_abs_predictions):
        test_order_position = int(np.where(overall_orders == test_id)[0])
        test_estimates = []
        for train_id in self.train_ids:
            train_order_position = int(np.where(overall_orders == train_id)[0])
            if test_order_position < train_order_position:
                # i.e.test is ranked higher than the top train -> higher activity than that of the top train sample
                test_estimates.append(Y_sign_and_abs_predictions[(test_id, train_id)][0] * 1
                                      + self.y_true_all[train_id])
                # No need of the below because:
                #     sv[(test_id, train_id)] = sv[(train_id, test_id)] = abs(y_train_id - y_test_id)
                # test_estimates.append(sv[(train_id, test_id)][0] * -1 + self.y_true[train_id])
            elif test_order_position > train_order_position:
                test_estimates.append(Y_sign_and_abs_predictions[(test_id, train_id)][0] * -1
                                      + self.y_true_all[train_id])
        return np.mean(test_estimates)

    def find_sum_of_estimates_of_top_x_tests(self, x, y_all_pred, Y_sign_and_abs_predictions=None):
        orders_overall = np.argsort(-y_all_pred)
        orders_tests = [i for i in orders_overall if i in self.test_ids]
        top_x_tests = orders_tests[:x]  # index of top 5 testids
        sum_y_pred = []
        sum_y_true = []

        if Y_sign_and_abs_predictions is None:  # i.e. not pairwise
            for test_id in top_x_tests:
                sum_y_pred.append(y_all_pred[test_id])
                sum_y_true.append(self.y_true_all[test_id])
        else:  # i.e.  pairwise
            for test_id in top_x_tests:
                sum_y_pred.append(
                    self.estimate_averaged_y_from_final_ranking(orders_overall, test_id, Y_sign_and_abs_predictions))
                sum_y_true.append(self.y_true_all[test_id])
        accuracy = mean_squared_error(sum_y_pred, sum_y_true)
        return sum(sum_y_pred), sum(sum_y_true), accuracy

    @staticmethod
    def calculate_correct_rate(correctly_identified_top_samples, true_top_samples):
        return len(correctly_identified_top_samples) / len(true_top_samples) if true_top_samples else np.nan
