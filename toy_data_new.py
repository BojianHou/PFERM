import matplotlib.pyplot as plt
import numpy as np
import time
from linear_ferm import Linear_FERM
from ferm import FERM
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP
from sklearn.model_selection import GridSearchCV
from collections import namedtuple

def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples + n_samples_low) + [-1] * (n_samples_low + n_samples)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [-1] * n_samples_low + [1] * n_samples_low + [-1] * n_samples
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    idx_A = list(range(0, n_samples+n_samples_low))
    idx_B = list(range(n_samples+n_samples_low, n_samples*2+n_samples_low*2))

    # print('(y, sensible_feature):')
    # for el in zip(y, sensible_feature):
    #     print(el)
    return X, y, sensible_feature_id, idx_A, idx_B


if __name__ == "__main__":
    start_time = time.perf_counter()
    print('start time is: ', start_time)

    pi = 5  # it means that the probability of female getting AD is the \pi times that of male
    print('-----------------PI is: {}-----------------'.format(pi))

    n_samples_low = 200  # number of males
    n_samples = pi * n_samples_low  # number of females
    X, y, sensible_feature, idx_A, idx_B = generate_toy_data(n_samples=n_samples,
                                                             n_samples_low=n_samples_low,
                                                             n_dimensions=2)
    # print('(y, sensible_feature):')
    # for el in zip(y, X[:, sensible_feature]):
    #     print(el)

    point_size = 150
    linewidth = 6

    step = 30
    alpha = 0.5
    plt.scatter(X[0:(n_samples+n_samples_low):step, 0], X[0:(n_samples+n_samples_low):step, 1], marker='o', s=point_size,
                c=y[0:(n_samples+n_samples_low):step], edgecolors='k', label='Group A', alpha=alpha)
    plt.scatter(X[(n_samples+n_samples_low)::step, 0], X[(n_samples+n_samples_low)::step, 1], marker='s', s=point_size,
                c=y[(n_samples+n_samples_low)::step], edgecolors='k', label='Group B', alpha=alpha)
    plt.legend()
    plt.title("Generated Dataset")
    plt.colorbar()
    plt.show()

    sensible_feature_values = sorted(list(set(X[:, sensible_feature])))
    dataXy = namedtuple('_', 'data, target')(X, y)

    # print('(y, sensible_feature):')
    # for el in zip(y, dataXy.data[:, sensible_feature]):
    #     print(el)

    C_list = [10 ** v for v in range(-6, 3)]
    print("List of C tested:", C_list)
    # Standard SVM -  Train an SVM using the training set
    # P as a prefix means prior knowledge
    # L as a prefix means linear
    # N as a prefix means nonlinear
    acc_list, mis_list, deo_list, \
    acc_LFERM_list, mis_LFERM_list, deo_LFERM_list, \
    acc_NLFERM_list, mis_NLFERM_list, deo_NLFERM_list, \
    acc_LPFERM_list, mis_LPFERM_list, deo_LPFERM_list, \
    acc_NLPFERM_list, mis_NLPFERM_list, deo_NLPFERM_list \
                = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    print('-----------------STANDARD LINEAR SVM-----------------')
    for c in C_list:
        print('C:', c)
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X, y)

        # Accuracy and Fairness
        pred = clf.predict(X)
        print('Accuracy:', accuracy_score(y, pred))
        # Fairness measure
        EO_LSVM = equalized_odds_measure_TP(dataXy, clf, [sensible_feature], ylabel=1)
        DEO_LSVM = np.abs(EO_LSVM[sensible_feature][sensible_feature_values[0]] -
                        1/pi*EO_LSVM[sensible_feature][sensible_feature_values[1]])
        print('DEO:', DEO_LSVM)
        acc_list.append(accuracy_score(y, pred))
        mis_list.append(1.0 - accuracy_score(y, pred))
        deo_list.append(DEO_LSVM)

    # Linear FERM
    print('-----------------LINEAR FERM-----------------')
    for c in C_list:
        print('C:', c)
        list_of_sensible_feature = X[:, sensible_feature]
        clf = svm.SVC(kernel='linear', C=c)
        algorithm = Linear_FERM(dataXy, clf, X[:, sensible_feature])
        algorithm.fit()

        # Accuracy and Fairness
        pred = algorithm.predict(X)
        print('Accuracy fair Linear FERM:', accuracy_score(y, pred))
        # Fairness measure
        EO_LFERM = equalized_odds_measure_TP(dataXy, algorithm, [sensible_feature], ylabel=1)
        DEO_LFERM = np.abs(EO_LFERM[sensible_feature][sensible_feature_values[0]] -
                        1/pi*EO_LFERM[sensible_feature][sensible_feature_values[1]])
        print('DEO fair Linear FERM:', DEO_LFERM)
        acc_LFERM_list.append(accuracy_score(y, pred))
        mis_LFERM_list.append(1.0 - accuracy_score(y, pred))
        deo_LFERM_list.append(DEO_LFERM)

    # Example of using FERM
    print('-----------------NONLINEAR FERM-----------------')
    # C = 0.1
    for C in C_list:
        gamma = 0.1
        print('\nTraining (non linear) FERM for C', C, 'and RBF gamma', gamma)
        clf = FERM(sensible_feature=dataXy.data[:, sensible_feature], C=C, gamma=gamma, kernel='rbf')
        clf.fit(dataXy.data, dataXy.target)

        # Accuracy and Fairness
        y_predict = clf.predict(dataXy.data)
        pred_train = clf.predict(dataXy.data)
        print('Accuracy fair (non linear) FERM:', accuracy_score(dataXy.target, pred_train))
        # Fairness measure
        EO_NLFERM = equalized_odds_measure_TP(dataXy, clf, [sensible_feature], ylabel=1)
        DEO_NLFERM = np.abs(EO_NLFERM[sensible_feature][sensible_feature_values[0]] -
                            1/pi*EO_NLFERM[sensible_feature][sensible_feature_values[1]])
        print('DEO fair (non linear) FERM:', DEO_NLFERM)
        acc_NLFERM_list.append(accuracy_score(y, pred_train))
        mis_NLFERM_list.append(1.0 - accuracy_score(y, pred_train))
        deo_NLFERM_list.append(DEO_NLFERM)

    # Linear PFERM
    print('-----------------LINEAR PFERM-----------------')
    for c in C_list:
        print('C:', c)
        list_of_sensible_feature = X[:, sensible_feature]
        clf = svm.SVC(kernel='linear', C=c)
        algorithm = Linear_FERM(dataXy, clf, X[:, sensible_feature], prior=True, pi=pi)
        algorithm.fit()

        # Accuracy and Fairness
        pred = algorithm.predict(X)
        print('Accuracy fair Linear PFERM:', accuracy_score(y, pred))
        # Fairness measure
        EO_LPFERM = equalized_odds_measure_TP(dataXy, algorithm, [sensible_feature], ylabel=1)
        DEO_LPERFM = np.abs(EO_LPFERM[sensible_feature][sensible_feature_values[0]] -
                        1/pi*EO_LPFERM[sensible_feature][sensible_feature_values[1]])
        print('DEO fair Linear PFERM:', DEO_LPERFM)
        acc_LPFERM_list.append(accuracy_score(y, pred))
        mis_LPFERM_list.append(1.0 - accuracy_score(y, pred))
        deo_LPFERM_list.append(DEO_LPERFM)

    # Nonlinear PFERM
    print('-----------------NONLINEAR PFERM-----------------')
    # C = 0.1
    for C in C_list:
        gamma = 0.1
        print('\nTraining (non linear) PFERM for C', C, 'and RBF gamma', gamma)
        clf = FERM(sensible_feature=dataXy.data[:, sensible_feature], C=C, gamma=gamma,
                   kernel='rbf', prior=True, pi=pi)
        clf.fit(dataXy.data, dataXy.target)

        # Accuracy and Fairness
        y_predict = clf.predict(dataXy.data)
        pred_train = clf.predict(dataXy.data)
        print('Accuracy fair (non linear) PFERM:', accuracy_score(dataXy.target, pred_train))
        # Fairness measure
        EO_NLPFERM = equalized_odds_measure_TP(dataXy, clf, [sensible_feature], ylabel=1)
        DEO_NLPFERM = np.abs(EO_NLPFERM[sensible_feature][sensible_feature_values[0]] -
                            1/pi*EO_NLPFERM[sensible_feature][sensible_feature_values[1]])

        print('DEO fair (non linear) PFERM:', DEO_NLPFERM)
        acc_NLPFERM_list.append(accuracy_score(y, pred_train))
        mis_NLPFERM_list.append(1.0 - accuracy_score(y, pred_train))
        deo_NLPFERM_list.append(DEO_NLPFERM)

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

    acc_matrix = np.vstack([acc_list, acc_LFERM_list, acc_LPFERM_list,
                            acc_NLFERM_list, acc_NLPFERM_list]).transpose()
    deo_matrix = np.vstack([deo_list, deo_LFERM_list, deo_LPFERM_list,
                            deo_NLFERM_list, deo_NLPFERM_list]).transpose()

    e = int(time.perf_counter() - start_time)
    print('Elapsed Time: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
