import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score


def equalized_odds(predictions, truth, sensitive_features):
    # measure the difference between the true positive rates of different groups
    # print(predictions)
    #predictions = predictions.argmax(1)

    group_true_pos_r = []
    values_of_sensible_feature = np.unique(sensitive_features)

    true_positive = np.sum([1.0 if predictions[i] == 1 and truth[i] == 1
                             else 0.0 for i in range(len(predictions))])
    all_positive = np.sum([1.0 if truth[i] == 1 else 0.0 for i in range(len(predictions))])
    all_true_pos_r = true_positive / all_positive

    for val in values_of_sensible_feature:
        positive_sensitive = np.sum([1.0 if sensitive_features[i] == val and truth[i] == 1 else 0.0
                                     for i in range(len(predictions))])
        if positive_sensitive > 0:
            true_positive_sensitive = np.sum([1.0 if predictions[i] == 1 and
                        sensitive_features[i] == val and truth[i] == 1
                         else 0.0 for i in range(len(predictions))])
            eq_tmp = true_positive_sensitive / positive_sensitive  # true positive rate
            group_true_pos_r.append(eq_tmp)

    return np.mean(np.abs(all_true_pos_r - group_true_pos_r))


def equalized_odds_measure_TP(data, model, sensitive_features, ylabel=1, rev_pred=1):
    '''
    True positive label for the groups defined by the values of the "sensible_features",
    with respect to the "model" on the "data".
    :param data: the data where to evaluate the True Positive Rate (Equal Opportunity).
    :param model:  the model that has to be evaluated.
    :param sensitive_features: the features used to split the data in groups.
    :param ylabel: the POSITIVE label (usually +1).
    :param rev_pred: an option to reverse the outputs of our model.
    :return: a dictionary with keys the sensitive_features and values dictionaries containing the True Positive Rates
    of the different subgroups of the specific sensitive feature.
    '''
    predictions = model.predict(data.data) * rev_pred
    truth = data.target
    eq_dict = {}
    for feature in sensitive_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if data.data[i, feature] == val and truth[i] == ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val and truth[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive  # true positive rate
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


def equalized_odds_measure_TP2(predictions, X, y, sensitive_features, ylabel=1):
    '''
    True positive label for the groups defined by the values of the "sensible_features",
    with respect to the "model" on the "data".
    :param predictions: the predictions for X
    :param y: the ground truth of X.
    :param sensitive_features: the features used to split the data in groups.
    :param ylabel: the POSITIVE label (usually +1).
    :return: a dictionary with keys the sensitive_features and values dictionaries containing the True Positive Rates
    of the different subgroups of the specific sensitive feature.
    '''

    eq_dict = {}
    for feature in sensitive_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(X[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if X[i, feature] == val and y[i] == ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and X[i, feature] == val and y[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive  # true positive rate
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


def demographic_parity_measure(predictions, X, sensitive_features):

    eq_dict = {}
    for feature in sensitive_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(X[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            sensitive = np.sum([1.0 if X[i, feature] == val else 0.0
                                         for i in range(len(predictions))])
            positive_sensitive = np.sum([1.0 if X[i, feature] == val and predictions[i] == 1 else 0.0
                                         for i in range(len(predictions))])
            if sensitive > 0:
                eq_tmp = positive_sensitive / sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature

    return eq_dict


def result_calculation(y_train, pred_train, y_test, pred_test):

    train_acc = accuracy_score(y_train, pred_train)
    train_bacc = balanced_accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    test_bacc = balanced_accuracy_score(y_test, pred_test)
    mis = 1.0 - test_bacc

    return train_acc, train_bacc, test_acc, test_bacc, mis


def print_results(C_list, train_acc_list, train_bacc_list, test_acc_list, test_bacc_list, deo_list):

    print('hyperpram C:', C_list)
    print('train_ACC  :', ["{:.4f}".format(acc) for acc in train_acc_list])
    print('train_BACC :', ["{:.4f}".format(bacc) for bacc in train_bacc_list])
    print('test_ACC   :', ["{:.4f}".format(acc) for acc in test_acc_list])
    print('test_BACC  :', ["{:.4f}".format(bacc) for bacc in test_bacc_list])
    print('PDEO       :', ["{:.4f}".format(deo) for deo in deo_list])


def print_results_single(train_acc, train_bacc, test_acc, test_bacc, deo, ddp):

    # print('train_ACC  : ', "{:.4f}".format(train_acc))
    # print('train_BACC : ', "{:.4f}".format(train_bacc))
    print('test_ACC: ', "{:.4f}".format(test_acc))
    # print('test_BACC  : ', "{:.4f}".format(test_bacc))
    print('DEO     : ', "{:.4f}".format(deo))
    print('DDP     : ', "{:.4f}".format(ddp))


def evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi=1):

    sensible_feature_values = sorted(list(set(X_train[:, sensible_feature_idx])))

    # Accuracy and Fairness
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    train_acc, train_bacc, test_acc, test_bacc, mis \
        = result_calculation(y_train, pred_train, y_test, pred_test)
    # Fairness measure
    EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
    # DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
    #              1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])
    DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                 EO[sensible_feature_idx][sensible_feature_values[1]])

    DP = demographic_parity_measure(pred_test, X_test, [sensible_feature_idx])
    DDP = np.abs(DP[sensible_feature_idx][sensible_feature_values[0]] -
                 DP[sensible_feature_idx][sensible_feature_values[1]])

    print_results_single(train_acc, train_bacc, test_acc, test_bacc, DEO, DDP)

    return train_acc, train_bacc, test_acc, test_bacc, DEO, DDP