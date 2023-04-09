from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def draw_pie(counts_list, labels, ROI='All AV45'):
    # ROI is the range of interest
    # labels is the list of all items, such as ['CN', 'MCI', 'AD']
    # counts_list is the count list of all items

    color_all = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                 'tab:olive', 'tab:cyan']
    colors = [color_all[i] for i in range(len(labels))]
    explode = [0.02] * len(labels)  # the distance between each pieces

    plt.pie(counts_list, labels=labels, colors=colors,
            startangle=90,
            explode=explode,
            autopct='%1.2f%%')

    plt.axis('equal')
    plt.title('{} in {}'.format(' vs '.join(labels), ROI))
    plt.savefig('figures/{} in {}.png'.format(' vs '.join(labels), ROI),
                bbox_inches='tight')
    plt.show()


def make_error_boxes(ax, x, y, x_min, x_max, y_min, y_max, method, facecolor='pink',
                     edgecolor='none', alpha=0.3):
    ax.plot(x, y, marker="o", markersize=5, markeredgecolor="none", markerfacecolor=facecolor)
    ax.add_patch(Rectangle((x_min, y_min), x_max, y_max, facecolor=facecolor, alpha=alpha,
                           edgecolor=edgecolor, label=method))
    artists = ax.errorbar(x, y, xerr=x - x_min, yerr=y - y_min,
                          fmt='none', ecolor=facecolor, alpha=2 * alpha)
    return artists


def plot_box_bak(data, result_mean, result_std, is_linear=True, y_axis='DEO'):
    # Create figure and axes
    # y_axis can be DEO or DDP, which refers to difference between equalized odds and demographic parity
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 6)
    facecolor = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray', 'tab:purple', 'tab:red']
    Method = ["Linear SVM", "Linear FERM", "Linear PFERM",
              "NonLinear SVM", "NonLinear FERM", "NonLinear PFERM"]

    # for i in range(len(Method)):
    if is_linear:
        methods = [0, 1, 2]
        title_name = "Accuracy vs {} Linear on {} Data".format(y_axis, data)
        file_name = "./figures/accuracy_vs_{}_{}_linear.png".format(y_axis, data)
    else:
        methods = [3, 4, 5]
        title_name = "Accuracy vs {} Nonlinear on {} Data".format(y_axis, data)
        file_name = "./figures/results_vs_{}_{}_nonlinear.png".format(y_axis, data)
    for i in methods:
        x = float(result_mean["ACC"][i])
        y = float(result_mean[y_axis][i])
        x_min = x - float(result_std["ACC"][i])
        x_max = 2 * float(result_std["ACC"][i])
        y_min = y - float(result_std[y_axis][i])
        y_max = 2 * float(result_std[y_axis][i])

        _ = make_error_boxes(ax, x, y, x_min, x_max, y_min, y_max, Method[i], facecolor=facecolor[i])

    plt.xlabel("Accuracy", fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.title(title_name, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14)
    plt.savefig(file_name, dpi=100)
    plt.show()


def plot_box(data, result_mean, result_std, is_linear=True, y_axis='DEO'):
    # Create figure and axes
    # y_axis can be DEO or DDP, which refers to difference between equalized odds and demographic parity
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 6)
    facecolor = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray', 'tab:purple', 'tab:red']

    Method = ["SVM", "FERM", "PFERM"]
    # for i in range(len(Method)):
    if is_linear:
        title_name = "Accuracy vs {} Linear on {} Data".format(y_axis, data)
        file_name = "./figures/accuracy_vs_{}_{}_linear.png".format(y_axis, data)
    else:
        title_name = "Accuracy vs {} Nonlinear on {} Data".format(y_axis, data)
        file_name = "./figures/results_vs_{}_{}_nonlinear.png".format(y_axis, data)
    for i in range(3):
        x = float(result_mean["ACC"][i])
        y = float(result_mean[y_axis][i])
        x_min = x - float(result_std["ACC"][i])
        x_max = 2 * float(result_std["ACC"][i])
        y_min = y - float(result_std[y_axis][i])
        y_max = 2 * float(result_std[y_axis][i])

        _ = make_error_boxes(ax, x, y, x_min, x_max, y_min, y_max, Method[i], facecolor=facecolor[i])

    plt.xlabel("Accuracy", fontsize=16)
    plt.ylabel(y_axis, fontsize=16)
    plt.title(title_name, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14)
    plt.savefig(file_name, dpi=100)
    plt.show()