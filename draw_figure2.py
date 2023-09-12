import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

dataset = 'toy_new' # 'adult' 'av45' 'toy_new' 'tadpole' 'toy_3'

for constraint in ['EO', 'DP']:
    for lamda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with open('./results/results_all_{}_constraint_{}_lamda_{}.pkl'
                          .format(dataset, constraint, lamda), 'rb') as f:
            result_list = pkl.load(f)

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
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.title('PI vs {} with constraint {} lamda {}'.format(measure, constraint, lamda), fontsize=16)
            plt.savefig('./figures/PI_vs_{}_constraint_{}_lamda_{}.png'.format(measure, constraint, lamda))
            plt.show()
            plt.close()