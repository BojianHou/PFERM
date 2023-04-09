import matplotlib.pyplot as plt
import numpy as np
import time
from load_data import load_dataset
from linear_ferm import Linear_FERM
from ferm import FERM, PFERM
from sklearn import svm
from measures import evaluate
from sklearn.model_selection import GridSearchCV
from collections import namedtuple
from plot import plot_box
import pickle as pkl


def train_test(X_train, X_test, y_train, y_test, sensible_feature_idx, pi, is_linear=False):

    if is_linear:
        kernel = 'linear'
        print('\n------------------------------Linear-------------------------------')
    else:
        kernel = 'rbf'
        print('\n----------------------------Non Linear-----------------------------')

    param_grid = [{'C': [0.01, 0.1, 1, 10.0],
                     'gamma': [0.1, 0.01],
                     'kernel': [kernel]}]

    print('Grid search for SVM...')
    svc = svm.SVC(kernel=kernel)
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator:', clf.best_estimator_)
    train_acc_SVM, train_bacc_SVM, test_acc_SVM, test_bacc_SVM, DEO_SVM, DDP_SVM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    print('Grid search for FERM...')
    algorithm = PFERM(sensible_feature=X_train[:, sensible_feature_idx], kernel=kernel, prior=False)
    clf = GridSearchCV(algorithm, param_grid, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator: FERM(C={}, gamma={})'.
          format(clf.best_estimator_.C, clf.best_estimator_.gamma))
    train_acc_FERM, train_bacc_FERM, test_acc_FERM, test_bacc_FERM, DEO_FERM, DDP_FERM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    print('Grid search for PFERM...')
    algorithm = PFERM(sensible_feature=X_train[:, sensible_feature_idx], kernel=kernel, prior=True, pi=pi)
    clf = GridSearchCV(algorithm, param_grid, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator: PFERM(C={}, gamma={})'.
          format(clf.best_estimator_.C, clf.best_estimator_.gamma))
    train_acc_PFERM, train_bacc_PFERM, test_acc_PFERM, test_bacc_PFERM, DEO_PFERM, DDP_PFERM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    return test_acc_SVM, test_acc_FERM, test_acc_PFERM, \
           DEO_SVM, DEO_FERM, DEO_PFERM, \
           DDP_SVM, DDP_FERM, DDP_PFERM


def main_bak(dataset, seed=42):

    # pi means that the probability of female getting AD is the pi times that of male
    X_train, X_test, y_train, y_test, sensible_feature_idx, pi = load_dataset(dataset, seed)

    param_grid_nonlinear = [{'C': [0.01, 0.1, 1, 10.0],
                            'gamma': [0.1, 0.01],
                            'kernel': ['rbf']}]
    param_grid_linear = [{'C': [0.01, 0.1, 1, 10.0],
                         'kernel': ['linear']}]

    print('\n----------------------------Linear-----------------------------')
    print('Grid search for Linear SVM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_linear, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator:', clf.best_estimator_)
    train_acc_LSVM, train_bacc_LSVM, test_acc_LSVM, test_bacc_LSVM, DEO_LSVM, DDP_LSVM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    print('Grid search for Linear FERM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_linear, n_jobs=1)
    dataset_train = namedtuple('_', 'data, target')(X_train, y_train)
    algorithm = Linear_FERM(dataset_train, clf, dataset_train.data[:, sensible_feature_idx])
    algorithm.fit()
    print('Best Estimator:', algorithm.model.best_estimator_)
    train_acc_LFERM, train_bacc_LFERM, test_acc_LFERM, test_bacc_LFERM, DEO_LFERM, DDP_LFERM \
        = evaluate(X_train, X_test, y_train, y_test, algorithm, sensible_feature_idx, pi)

    print('Grid search for Linear PFERM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_linear, n_jobs=1)
    dataset_train = namedtuple('_', 'data, target')(X_train, y_train)
    algorithm = Linear_FERM(dataset_train, clf, dataset_train.data[:, sensible_feature_idx], prior=True, pi=pi)
    algorithm.fit()
    print('Best Estimator:', algorithm.model.best_estimator_)
    train_acc_LPFERM, train_bacc_LPFERM, test_acc_LPFERM, test_bacc_LPFERM, DEO_LPFERM, DDP_LPFERM \
        = evaluate(X_train, X_test, y_train, y_test, algorithm, sensible_feature_idx, pi)

    print('\n----------------------------NonLinear-----------------------------')
    print('Grid search for NonLinear SVM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_nonlinear, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator:', clf.best_estimator_)
    train_acc_NLSVM, train_bacc_NLSVM, test_acc_NLSVM, test_bacc_NLSVM, DEO_NLSVM, DDP_NLSVM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    print('Grid search for NonLinear FERM...')
    algorithm = PFERM(sensible_feature=X_train[:, sensible_feature_idx])
    clf = GridSearchCV(algorithm, param_grid_nonlinear, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator: Nonlinear FERM(C={}, gamma={})'.
          format(clf.best_estimator_.C, clf.best_estimator_.gamma))
    train_acc_NLFERM, train_bacc_NLFERM, test_acc_NLFERM, test_bacc_NLFERM, DEO_NLFERM, DDP_NLFERM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    print('Grid search for NonLinear PFERM...')
    algorithm = PFERM(sensible_feature=X_train[:, sensible_feature_idx], prior=True, pi=pi)
    clf = GridSearchCV(algorithm, param_grid_nonlinear, n_jobs=1)
    clf.fit(X_train, y_train)
    print('Best Estimator: Nonlinear PFERM(C={}, gamma={})'.
          format(clf.best_estimator_.C, clf.best_estimator_.gamma))
    train_acc_NLPFERM, train_bacc_NLPFERM, test_acc_NLPFERM, test_bacc_NLPFERM, DEO_NLPFERM, DDP_NLPFERM \
        = evaluate(X_train, X_test, y_train, y_test, clf, sensible_feature_idx, pi)

    return test_acc_LSVM, test_acc_LFERM, test_acc_LPFERM, \
           test_acc_NLSVM, test_acc_NLFERM, test_acc_NLPFERM, \
           DEO_LSVM, DEO_LFERM, DEO_LPFERM, \
           DEO_NLSVM, DEO_NLFERM, DEO_NLPFERM, \
           DDP_LSVM, DDP_LFERM, DDP_LPFERM, \
           DDP_NLSVM, DDP_NLFERM, DDP_NLPFERM


def main(dataset, is_linear=False, pi=2):

    print('=================================PI: {}=================================='.format(pi))

    test_acc_SVM_list, test_acc_FERM_list, test_acc_PFERM_list, \
    DEO_SVM_list, DEO_FERM_list, DEO_PFERM_list, \
    DDP_SVM_list, DDP_FERM_list, DDP_PFERM_list, \
        = [], [], [], [], [], [], [], [], []

    for seed in [0, 42, 66, 666, 777]:
        print('\n========================seed {}========================'.format(seed))
        # pi means that the probability of female getting AD is the pi times that of male
        X_train, X_test, y_train, y_test, sensible_feature_idx, pi = load_dataset(dataset, seed, pi=pi)

        test_acc_SVM, test_acc_FERM, test_acc_PFERM, \
        DEO_SVM, DEO_FERM, DEO_PFERM, \
        DDP_SVM, DDP_FERM, DDP_PFERM \
            = train_test(X_train, X_test, y_train, y_test, sensible_feature_idx, pi, is_linear=is_linear)

        test_acc_SVM_list.append(test_acc_SVM)
        test_acc_FERM_list.append(test_acc_FERM)
        test_acc_PFERM_list.append(test_acc_PFERM)
        DEO_SVM_list.append(DEO_SVM), DEO_FERM_list.append(DEO_FERM), DEO_PFERM_list.append(DEO_PFERM)
        DDP_SVM_list.append(DDP_SVM), DDP_FERM_list.append(DDP_FERM), DDP_PFERM_list.append(DDP_PFERM)

    result_mean, result_std = {}, {}
    result_mean['ACC'] = [np.mean(test_acc_SVM_list), np.mean(test_acc_FERM_list), np.mean(test_acc_PFERM_list)]
    result_mean['DEO'] = [np.mean(DEO_SVM_list), np.mean(DEO_FERM_list), np.mean(DEO_PFERM_list)]
    result_mean['DDP'] = [np.mean(DDP_SVM_list), np.mean(DDP_FERM_list), np.mean(DDP_PFERM_list)]
    result_std['ACC'] = [np.std(test_acc_SVM_list), np.std(test_acc_FERM_list), np.std(test_acc_PFERM_list)]
    result_std['DEO'] = [np.std(DEO_SVM_list), np.std(DEO_FERM_list), np.std(DEO_PFERM_list)]
    result_std['DDP'] = [np.std(DDP_SVM_list), np.std(DDP_FERM_list), np.std(DDP_PFERM_list)]

    plot_box(dataset, result_mean, result_std, is_linear=is_linear, y_axis='DEO')
    plot_box(dataset, result_mean, result_std, is_linear=is_linear, y_axis='DDP')

    print('-----SVM------')
    print('ACC Mean±Std {:.4f}±{:.4f}'.format(np.mean(test_acc_SVM_list), np.std(test_acc_SVM_list)))
    print('DEO Mean±Std {:.4f}±{:.4f}'.format(np.mean(DEO_SVM_list), np.std(DEO_SVM_list)))
    print('DDP Mean±Std {:.4f}±{:.4f}'.format(np.mean(DDP_SVM_list), np.std(DDP_SVM_list)))
    print('-----FERM------')
    print('ACC Mean±Std {:.4f}±{:.4f}'.format(np.mean(test_acc_FERM_list), np.std(test_acc_FERM_list)))
    print('DEO Mean±Std {:.4f}±{:.4f}'.format(np.mean(DEO_FERM_list), np.std(DEO_FERM_list)))
    print('DDP Mean±Std {:.4f}±{:.4f}'.format(np.mean(DDP_FERM_list), np.std(DDP_FERM_list)))
    print('-----PFERM------')
    print('ACC Mean±Std {:.4f}±{:.4f}'.format(np.mean(test_acc_PFERM_list), np.std(test_acc_PFERM_list)))
    print('DEO Mean±Std {:.4f}±{:.4f}'.format(np.mean(DEO_PFERM_list), np.std(DEO_PFERM_list)))
    print('DDP Mean±Std {:.4f}±{:.4f}'.format(np.mean(DDP_PFERM_list), np.std(DDP_PFERM_list)))

    result = {'mean': result_mean, 'std': result_std}
    with open('./results/result_{}_{}.pkl'.format(dataset, pi), 'wb') as f:
        pkl.dump(result, f)

    return result


if __name__ == "__main__":
    start_time = time.perf_counter()
    print('start time is: ', start_time)

    # point_size = 150
    # linewidth = 6
    # step = 30
    # alpha = 0.5

    dataset = 'toy_new'  # 'adult' 'av45' 'toy_new' 'tadpole' 'toy_3'

    result_list = []
    for pi in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        result = main(dataset, is_linear=False, pi=pi)
        result_list.append(result)

    with open('./results/results_all_{}.pkl'.format(dataset), 'wb') as f:
        pkl.dump(result_list, f)

    x = np.linspace(1, 10, 10)
    measurements = ['ACC', 'DEO', 'DDP']
    methods = ['SVM', 'FERM', 'PFERM']
    for measure in measurements:
        for i in range(3):
            plt.errorbar(x, [result['mean'][measure][i] for result in result_list],
                         [result['std'][measure][i] for result in result_list], marker='.', label=methods[i])
            plt.xlabel('PI', fontsize=16)
            plt.ylabel(measure, fontsize=16)
        plt.legend()
        plt.savefig('./figures/PI_vs_{}_constraint_EO.png'.format(measure))
        plt.show()
        plt.close()

    # acc_matrix = np.vstack([test_bacc_list, test_bacc_LFERM_list, test_bacc_LPFERM_list,
    #                         test_bacc_NLFERM_list, test_bacc_NLPFERM_list]).transpose()
    # deo_matrix = np.vstack([deo_list, deo_LFERM_list, deo_LPFERM_list,
    #                         deo_NLFERM_list, deo_NLPFERM_list]).transpose()

    e = int(time.perf_counter() - start_time)
    print('Elapsed Time: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))