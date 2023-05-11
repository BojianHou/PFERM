import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

dataset = 'toy_new' # 'adult' 'av45' 'toy_new' 'tadpole' 'toy_3'

for constraint in ['EO', 'DP']:
    result_list = []
    for pi in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        with open('./results/result_{}_{}_constraint_{}.pkl'.format(dataset, pi, constraint), 'rb') as f:
            result = pkl.load(f)
        result_list.append(result)

    # with open('./results/results_all_{}.pkl'.format(dataset), 'rb') as f:
    #     result_list = pkl.load(f)

    x = np.linspace(1, 9, 9)
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

        plt.title('PI vs {} with constraint {}'.format(measure, constraint), fontsize=16)
        plt.savefig('./figures/PI_vs_{}_constraint_{}.png'.format(measure, constraint))
        plt.show()
        plt.close()