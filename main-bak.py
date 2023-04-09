import matplotlib.pyplot as plt
import numpy as np
import time
from load_data import load_dataset
from linear_ferm import Linear_FERM
from ferm import FERM
from sklearn import svm
from sklearn.metrics import accuracy_score
from measures import equalized_odds_measure_TP2, result_calculation, print_results
from sklearn.model_selection import GridSearchCV
from collections import namedtuple


if __name__ == "__main__":
    start_time = time.perf_counter()
    print('start time is: ', start_time)

    point_size = 150
    linewidth = 6
    step = 30
    alpha = 0.5

    dataset = 'tadpole'
    X_train, X_val, X_test, y_train, y_val, y_test, \
    sensible_feature_idx = load_dataset(dataset)

    sensible_feature_values = sorted(list(set(X_train[:, sensible_feature_idx])))
    train_Xy = namedtuple('_', 'data, target')(X_train, y_train)
    test_Xy = namedtuple('_', 'data, target')(X_test, y_test)
    pi = 2  # it means that the probability of female getting AD is the pi times that of male
    # print('(y, sensible_feature):')
    # for el in zip(y_train, train_Xy.data[:, sensible_feature_idx]):
    #     print(el)

    C_list = [10 ** v for v in range(-3, 3)]
    gamma_list = [10 ** v for v in range(-3, 3)]
    print("List of C tested:", C_list)

    # P as a prefix means prior knowledge
    # L as a prefix means linear
    # N as a prefix means nonlinear
    train_acc_list, test_acc_list, train_bacc_list, test_bacc_list, mis_list, deo_list, \
    train_acc_LFERM_list, test_acc_LFERM_list, train_bacc_LFERM_list, test_bacc_LFERM_list, mis_LFERM_list, deo_LFERM_list, \
    train_acc_NLFERM_list, test_acc_NLFERM_list, train_bacc_NLFERM_list, test_bacc_NLFERM_list, mis_NLFERM_list, deo_NLFERM_list, \
    train_acc_LPFERM_list, test_acc_LPFERM_list, train_bacc_LPFERM_list, test_bacc_LPFERM_list, mis_LPFERM_list, deo_LPFERM_list, \
    train_acc_NLPFERM_list, test_acc_NLPFERM_list, train_bacc_NLPFERM_list, test_bacc_NLPFERM_list, mis_NLPFERM_list, deo_NLPFERM_list \
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Standard SVM -  Train an SVM using the training set
    print('-----------------STANDARD LINEAR SVM-----------------')
    for c in C_list:
        # print('C:', c)
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X_train, y_train)

        # Accuracy and Fairness
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)

        train_acc, train_bacc, test_acc, test_bacc, mis \
            = result_calculation(y_train, pred_train, y_test, pred_test)
        # Fairness measure
        EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
        DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                    1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_bacc_list.append(train_bacc)
        test_bacc_list.append(test_bacc)
        mis_list.append(mis)
        deo_list.append(DEO)

    print_results(C_list, train_acc_list, train_bacc_list, test_acc_list, test_bacc_list, deo_list)

    # Linear FERM
    print('-----------------LINEAR FERM-----------------')
    for c in C_list:
        # print('C:', c)
        list_of_sensible_feature = X_train[:, sensible_feature_idx]
        clf = svm.SVC(kernel='linear', C=c)
        algorithm = Linear_FERM(train_Xy, clf, X_train[:, sensible_feature_idx])
        algorithm.fit()

        # Accuracy and Fairness
        pred_train = algorithm.predict(X_train)
        pred_test = algorithm.predict(X_test)

        train_acc, train_bacc, test_acc, test_bacc, mis \
            = result_calculation(y_train, pred_train, y_test, pred_test)
        # Fairness measure
        EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
        DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                     1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])

        train_acc_LFERM_list.append(train_acc)
        test_acc_LFERM_list.append(test_acc)
        train_bacc_LFERM_list.append(train_bacc)
        test_bacc_LFERM_list.append(test_bacc)
        mis_LFERM_list.append(mis)
        deo_LFERM_list.append(DEO)

    print_results(C_list, train_acc_LFERM_list, train_bacc_LFERM_list,
                  test_acc_LFERM_list, test_bacc_LFERM_list, deo_LFERM_list)

    # Example of using FERM
    print('-----------------NONLINEAR FERM-----------------')
    # C = 0.1

    for gamma in gamma_list:
        for C in C_list:
            print('\nTraining (non linear) FERM for C', C, 'and RBF gamma', gamma)
            clf = FERM(sensible_feature=train_Xy.data[:, sensible_feature_idx], C=C, gamma=gamma, kernel='rbf')
            clf.fit(train_Xy.data, train_Xy.target)

            # Accuracy and Fairness
            pred_train = clf.predict(train_Xy.data)
            pred_test = clf.predict(test_Xy.data)

            train_acc, train_bacc, test_acc, test_bacc, mis \
                = result_calculation(y_train, pred_train, y_test, pred_test)
            # Fairness measure
            EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
            DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                         1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])

            train_acc_NLFERM_list.append(train_acc)
            test_acc_NLFERM_list.append(test_acc)
            train_bacc_NLFERM_list.append(train_bacc)
            test_bacc_NLFERM_list.append(test_bacc)
            mis_NLFERM_list.append(mis)
            deo_NLFERM_list.append(DEO)

        print_results(C_list, train_acc_NLFERM_list, train_bacc_NLFERM_list,
                      test_acc_NLFERM_list, test_bacc_NLFERM_list, deo_NLFERM_list)

    # Linear PFERM
    print('-----------------LINEAR PFERM-----------------')
    for c in C_list:
        print('C:', c)
        list_of_sensible_feature = X_train[:, sensible_feature_idx]
        clf = svm.SVC(kernel='linear', C=c)
        algorithm = Linear_FERM(train_Xy, clf, X_train[:, sensible_feature_idx], prior=True, pi=pi)
        algorithm.fit()

        # Accuracy and Fairness
        pred_train = algorithm.predict(X_train)
        pred_test = algorithm.predict(X_test)

        train_acc, train_bacc, test_acc, test_bacc, mis \
            = result_calculation(y_train, pred_train, y_test, pred_test)
        # Fairness measure
        EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
        DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                     1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])

        train_acc_LPFERM_list.append(train_acc)
        test_acc_LPFERM_list.append(test_acc)
        train_bacc_LPFERM_list.append(train_bacc)
        test_bacc_LPFERM_list.append(test_bacc)
        mis_LPFERM_list.append(mis)
        deo_LPFERM_list.append(DEO)

    print_results(C_list, train_acc_LPFERM_list, train_bacc_LPFERM_list,
                  test_acc_LPFERM_list, test_bacc_LPFERM_list, deo_LPFERM_list)

    # Nonlinear PFERM
    print('-----------------NONLINEAR PFERM-----------------')
    # C = 0.1
    for gamma in gamma_list:
        for C in C_list:
            print('\nTraining (non linear) PFERM for C', C, 'and RBF gamma', gamma)
            clf = FERM(sensible_feature=train_Xy.data[:, sensible_feature_idx], C=C, gamma=gamma,
                       kernel='rbf', prior=True, pi=pi)
            clf.fit(train_Xy.data, train_Xy.target)

            # Accuracy and Fairness
            pred_test = clf.predict(test_Xy.data)
            pred_train = clf.predict(train_Xy.data)
            train_acc, train_bacc, test_acc, test_bacc, mis \
                = result_calculation(y_train, pred_train, y_test, pred_test)
            # Fairness measure
            EO = equalized_odds_measure_TP2(pred_test, X_test, y_test, [sensible_feature_idx], ylabel=1)
            DEO = np.abs(EO[sensible_feature_idx][sensible_feature_values[0]] -
                         1 / pi * EO[sensible_feature_idx][sensible_feature_values[1]])

            train_acc_NLPFERM_list.append(train_acc)
            test_acc_NLPFERM_list.append(test_acc)
            train_bacc_NLPFERM_list.append(train_bacc)
            test_bacc_NLPFERM_list.append(test_bacc)
            mis_NLPFERM_list.append(mis)
            deo_NLPFERM_list.append(DEO)

        print_results(C_list, train_acc_NLPFERM_list, train_bacc_NLPFERM_list,
                      test_acc_NLPFERM_list, test_bacc_NLPFERM_list, deo_NLPFERM_list)

    plt.figure(2)
    plt.scatter(mis_list, deo_list, marker='o', s=point_size, edgecolors='k', label='Linear SVM', alpha=alpha)
    plt.scatter(mis_LFERM_list, deo_LFERM_list, marker='s', s=point_size, edgecolors='k', label='Linear FERM',
                alpha=alpha)
    plt.scatter(mis_NLFERM_list, deo_NLFERM_list, marker='^', s=point_size, edgecolors='k', label='NonLinear FERM',
                alpha=alpha)
    plt.scatter(mis_LPFERM_list, deo_LPFERM_list, marker='*', s=point_size, edgecolors='k', label='Linear PFERM',
                alpha=alpha)
    plt.scatter(mis_NLPFERM_list, deo_NLPFERM_list, marker='X', s=point_size, edgecolors='k', label='NonLinear PFERM',
                alpha=alpha)
    plt.legend()
    plt.xlabel('Misclassification Error')
    plt.ylabel('PDEO')
    # plt.title("LSVM vs LFERM vs NLFERM vs LPFERM vs NLPFERM (different C values)")
    plt.title("PDEO and MIS Error Comparisons with Different C Values (PI: {})".format(pi))
    plt.savefig('./figures/Comparison_PI_{}'.format(pi))
    plt.show()
    plt.close()

    # acc_matrix = np.vstack([test_bacc_list, test_bacc_LFERM_list, test_bacc_LPFERM_list,
    #                         test_bacc_NLFERM_list, test_bacc_NLPFERM_list]).transpose()
    # deo_matrix = np.vstack([deo_list, deo_LFERM_list, deo_LPFERM_list,
    #                         deo_NLFERM_list, deo_NLPFERM_list]).transpose()

    e = int(time.perf_counter() - start_time)
    print('Elapsed Time: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))