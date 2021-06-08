import  cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow.keras.backend as K

from io import StringIO
from matplotlib.font_manager import FontProperties

from data import compare_masks
from gradients_and_influence import avg_influence


def create_distribution(mean, covariance, n_samples, label, seed=None):
    np.random.seed(seed)
    x = np.random.multivariate_normal(mean, covariance, n_samples)
    y = np.zeros((n_samples, 1)) if label == 0 else np.ones((n_samples, label))
    return x, y


def create_summary(x, y, flipped, model, grad, hess):
    plot3d_points = np.array([[0, 0, 0, 0, 0, 0] for i in range(len(y))], dtype=np.float)
    for i in range(len(y)):
        plot3d_points[i][0] = x[i][0]
        plot3d_points[i][1] = x[i][1]
        plot3d_points[i][2] = y[i]
        plot3d_points[i][3] = model(x[None, i])
        plot3d_points[i][4] = avg_influence(i, grad, hess)
        plot3d_points[i][5] = flipped[i]
    return plot3d_points


def get_class_weights(labels):
    total_samples = len(labels)
    n_1s = np.sum(labels)
    n_0s = total_samples - n_1s

    weight_for_0 = (1 / n_0s) * (total_samples) / 2.0
    weight_for_1 = (1 / n_1s) * (total_samples) / 2.0

    return {0: weight_for_0, 1: weight_for_1}


def calculate_risk(y_true, y_pred):
    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)
    if type(y_true) != np.ndarray:
        y_true = np.array(y_true)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze()
    if len(y_true.shape) > 1:
        y_true = y_true.squeeze()
    # return (np.abs(y_pred-y_true) * get_sample_weights(y_true)).sum()
    return np.abs(y_pred-y_true).sum() / len(y_true)


def get_sample_weights(labels):
    if type(labels) != np.ndarray:
        labels = np.array(labels)
    if len(labels.shape) > 1:
        labels = labels.squeeze()
    n_samples = len(labels)
    n_samples_1 = labels.sum()
    n_samples_0 = n_samples - n_samples_1
    return (1 - labels) / n_samples_0 + labels / n_samples_1


# def flip_random_labels(y, probability):
#     flipped = np.zeros(y.shape, dtype=np.int)
#     y = np.copy(y)
#     for i in range(len(y)):
#         if random.random() < probability:
#             y[i] = 1 - y[i]
#             flipped[i] = 1
#     return y, flipped


def flip_random_labels(y, percentage):
    flipped = np.zeros(y.shape, dtype=np.int)
    y = np.copy(y)
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    n_class_1_flips = int(percentage * len(class_1_indices))

    class_0_flip_indices = np.random.choice(class_0_indices, size=n_class_1_flips, replace=False)
    y[class_0_flip_indices] = 1
    flipped[class_0_flip_indices] = 1

    class_1_flip_indices = np.random.choice(class_1_indices, size=n_class_1_flips, replace=False)
    y[class_1_flip_indices] = 0
    flipped[class_1_flip_indices] = 1

    return y, flipped


def plot2d(x, y, fig, subplot):
    ax = fig.add_subplot(subplot)
    for i in range(len(y)):
        plt.plot(x[i][0], x[i][1], 'o', color="black" if y[i] == 0 else "red", ms=0.5)
    min_x, max_x = np.min(x[:, 0]), np.max(x[:, 0])
    plt.xlim(min_x * 1.01, max_x * 1.01)
    plt.axis('equal')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')


def plot3d(summary, fig, subplot):
    ax = fig.add_subplot(subplot, projection='3d')
    for i in range(len(summary)):
        ax.scatter(summary[i][0], summary[i][1], summary[i][4],
                   c="red" if summary[i][2] == 1 else "black", s=0.5)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('influence')


def plot_decision_boundary(x, y, trained_model, fig, subplot, title=None, x_range=None, y_range=None):
    ax = fig.add_subplot(subplot)
    if x_range is None:
        min_x, max_x = np.min(x[:, 0]), np.max(x[:, 0])
    else:
        min_x, max_x = x_range
    if y_range is None:
        min_y, max_y = np.min(x[:, 1]), np.max(x[:, 1])
    else:
        min_y, max_y = y_range
    h = .02
    xx, yy = np.meshgrid(np.arange(min_x * 1.01, max_x * 1.01, h),
                         np.arange(min_y * 1.01, max_y * 1.01, h))

    plt.axis('equal')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    if title is not None:
        ax.set_title(title)

    Z = trained_model.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=1, cmap=plt.get_cmap("coolwarm"), alpha=.1, vmin=0.0, vmax=1.0)
    for i in range(len(y)):
        plt.plot(x[i][0], x[i][1], 'o', color="black" if y[i] == 0 else "red", ms=0.5)

    def boundary_function(x, trained_model):
        return (-(trained_model.layers[0].kernel[0] * x + trained_model.layers[0].bias)/trained_model.layers[0].kernel[1]).numpy()[0]

    ax.axline((min_x, boundary_function(min_x, trained_model)), (max_x, boundary_function(max_x, trained_model)),
              c="blue", linestyle="--")
    ax.set_aspect('auto')
    ax.set_xbound(min_x * 1.01, max_x * 1.01)
    ax.set_ybound(min_y * 1.01, max_y * 1.01)


def plot_probability_map(x, y, trained_classifier, fig, subplot, title=None, x_range=None, y_range=None):
    ax = fig.add_subplot(subplot)
    if x_range is None:
        min_x, max_x = np.min(x[:, 0]), np.max(x[:, 0])
    else:
        min_x, max_x = x_range
    if y_range is None:
        min_y, max_y = np.min(x[:, 1]), np.max(x[:, 1])
    else:
        min_y, max_y = y_range
    h = .02
    xx, yy = np.meshgrid(np.arange(min_x * 1.01, max_x * 1.01, h),
                         np.arange(min_y * 1.01, max_y * 1.01, h))
    plt.axis('equal')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    if title is not None:
        ax.set_title(title)

    Z = trained_classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].round(2)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=1, cmap=plt.get_cmap("coolwarm"), alpha=.1, vmin=0.0, vmax=1.0)
    scatter_points = np.where(Z == 0.5)
    ax.plot(xx[scatter_points], yy[scatter_points], color="gray", ms=0.5, alpha=0.1)

    for i in range(len(y)):
        plt.plot(x[i][0], x[i][1], 'o', color="black" if y[i] == 0 else "red", ms=0.5)


    ax.set_aspect('auto')
    ax.set_xbound(min_x * 1.01, max_x * 1.01)
    ax.set_ybound(min_y * 1.01, max_y * 1.01)


def get_experiments_dict(path_to_summary, path_to_parameters):

    with open(path_to_summary, "r") as results_summary:
        results_summary = results_summary.readlines()
    experiments = {}
    with open(path_to_parameters, "r") as parameters_summary:
        parameters_summary = parameters_summary.readlines()
    row = 1
    for line in results_summary:
        if line.startswith("Experiment"):
            if len(experiments) > 0:
                parameters_string = parameters_summary[0] + parameters_summary[row]
                experiments[name] = \
                    {"results": pd.read_csv(StringIO(dataframe_string), sep=","),
                     "parameters": pd.read_csv(StringIO(parameters_string), sep=",")}
                row += 1
            name = line.split(",")[0]
            experiments[name] = 0
            dataframe_string = ""
            continue
        dataframe_string += line

    parameters_string = parameters_summary[0] + parameters_summary[row]
    experiments[name] = \
        {"results": pd.read_csv(StringIO(dataframe_string), sep=","),
         "parameters": pd.read_csv(StringIO(parameters_string), sep=",")}

    return experiments


def plot_naive(x, y, plot_size=(420, 420)):
    canvas_0 = np.zeros((plot_size[0], plot_size[1]), dtype=np.float64)
    canvas_1 = np.zeros((plot_size[0], plot_size[1]), dtype=np.float64)
    plot = np.ones((plot_size[0], plot_size[1], 3), dtype=np.float64)

    # labels, counts = np.unique(y, return_counts=True)

    min_x1 = np.min(x[:, 0])
    max_x1 = np.max(x[:, 0])
    min_x2 = np.min(x[:, 1])
    max_x2 = np.max(x[:, 1])
    resolution_x1 = plot_size[0]/(max_x1 - min_x1)
    if resolution_x1 == np.inf:
        resolution_x1 = 0
    resolution_x2 = plot_size[1]/(max_x2 - min_x2)
    if resolution_x2 == np.inf:
        resolution_x2 = 0
    for i in range(len(y)):
        x_1, x_2 = x[i]
        label = y[i]
        x_1 = min(plot_size[0] - 1, max(0, int((x_1-min_x1) * resolution_x1)))
        x_2 = min(plot_size[1] - 1, max(0, plot_size[1] - int((x_2-min_x2) * resolution_x2)))
        # if label == 0:
        #     canvas[x_2 - 1: x_2 + 2, x_1, :] += [1, 0, 0]
        #     canvas[x_2, x_1 - 1: x_1 + 2, :] += [1, 0, 0]
        # else:
        #     canvas[x_2 - 1: x_2 + 2, x_1, :] += [0, 0, 1]
        #     canvas[x_2, x_1 - 1: x_1 + 2, :] += [0, 0, 1]
        if label == 0:
            canvas_0[x_2, x_1] += 1
        else:
            canvas_1[x_2, x_1] += 1

    canvas_0 /= np.max(canvas_0)
    canvas_1 /= np.max(canvas_1)
    x_m = 0.05
    canvas_0 = x_m + canvas_0*(1-x_m)
    canvas_1 = x_m + canvas_1*(1-x_m)
    class_0_pixels = np.where(canvas_0 > x_m)
    class_1_pixels = np.where(canvas_1 > x_m)
    plot[class_0_pixels[0], class_0_pixels[1]] = np.concatenate(
        [[[1] for i in range(len(class_0_pixels[0]))],
         1 - canvas_0[class_0_pixels[0], class_0_pixels[1]][..., None],
         1 - canvas_0[class_0_pixels[0], class_0_pixels[1]][..., None]],
        axis=-1
    )
    plot[class_1_pixels[0], class_1_pixels[1]] = np.concatenate(
        [1 - canvas_1[class_1_pixels[0], class_1_pixels[1]][..., None],
         1 - canvas_1[class_1_pixels[0], class_1_pixels[1]][..., None],
         [[1] for i in range(len(class_1_pixels[0]))]],
        axis=-1
    )

    return (255*plot).astype(np.uint8)



colors = [[0.4, 0.8, 0.8],
 [0.3, 0.1, 0.2],
 [0.1, 0.8, 0.5],
 [0.7, 0.2, 0.9],
 [0.6, 0.6, 0.1],
 [0.2, 0.5, 0.9],
 [0.4, 0.1, 0.0],
 [0.1, 0.6, 0.3],
 [0.5, 0.5, 0.1],
 [0.7, 0.8, 0.7]]


def plot_correction_f_for_ratio(experiments_dict, keys_of_interest):
    f, ax = plt.subplots()

    p_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["p"].values for key in keys_of_interest]))
    visited_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["visited_percentage"].values for key in keys_of_interest]))
    methods = np.unique(np.concatenate([experiments_dict[key]["parameters"]["method"].values for key in keys_of_interest]))
    for method in methods:
        for color_number, p_value in enumerate(p_values):
            x = []
            precision = []
            recall = []
            fm = []
            precision_e = []
            recall_e = []
            fm_e = []

            for key in keys_of_interest:
                if experiments_dict[key]["parameters"]["p"].values[0] == p_value and \
                        experiments_dict[key]["parameters"]["method"].values[0] == method:
                    x.append(experiments_dict[key]["parameters"]["visited_percentage"].values[0])
                    precision.append(np.mean(experiments_dict[key]["results"]["Precision"]))
                    precision_e.append(np.std(experiments_dict[key]["results"]["Precision"]))
                    recall.append(np.mean(experiments_dict[key]["results"]["Recall"]))
                    recall_e.append(np.std(experiments_dict[key]["results"]["Recall"]))
                    fm.append(np.mean(experiments_dict[key]["results"]["F-measure"]))
                    fm_e.append(np.std(experiments_dict[key]["results"]["F-measure"]))

            x = [0] + x
            precision = [0] + precision
            precision_e = [0] + precision_e
            recall = [0] + recall
            recall_e = [0] + recall_e
            fm = [0] + fm
            fm_e = [0] + fm_e
            ax.errorbar(x, fm, fm_e, color=colors[color_number], capsize=5,
                        capthick=2 if method == "relabel" else None,
                        linestyle='solid' if method == "relabel" else '-.',
                        label="M: {}, P change: {}".format(method, p_value))
            # ax.errorbar(x, precision, precision_e, color=colors[color_number], linestyle="-.", capsize=5, label="Precision")
            # ax.errorbar(x, recall, recall_e, color=colors[color_number], linestyle="--", capsize=5, label="Recall")
    plt.xticks([0] + list(visited_values))
    ax.set_xlim(xmin=-0.01, xmax=max(visited_values) + 0.01)
    ax.set_ylim(ymin=0, ymax=1)
    ax.set(xlabel='Ratio of samples visited at first iteration', ylabel='F-measure')
    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(loc='upper left', ncol=len(methods), prop=fontP)
    plt.title("Label correction F-measure\n(for data with {}% of class 0 samples)".
              format(100*experiments_dict[keys_of_interest[0]]["parameters"]["ratio"].values[0]))
    return f


def plot_f_for_ratio(experiments_dict, keys_of_interest):
    f, [ax2, ax] = plt.subplots(2, 1, sharex=True)
    f.set_size_inches(7, 8.75)
    f_benchmark = np.concatenate([experiments_dict[key]["results"]["F-measure original"].values for key in keys_of_interest])
    f_benchmark_mean = np.mean(f_benchmark)
    f_benchmark_std = np.std(f_benchmark)
    p_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["p"].values for key in keys_of_interest]))
    visited_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["visited_percentage"].values for key in keys_of_interest]))
    methods = np.unique(np.concatenate([experiments_dict[key]["parameters"]["method"].values for key in keys_of_interest]))
    min_corrected_f = 1.0
    max_corrected_f = 0.0
    for method in methods:
        for color_number, p_value in enumerate(p_values):
            x = []
            y = []
            e = []
            zero_iterations_data = []
            for key in keys_of_interest:
                if experiments_dict[key]["parameters"]["p"].values[0] == p_value and \
                        experiments_dict[key]["parameters"]["method"].values[0] == method:
                    x.append(experiments_dict[key]["parameters"]["visited_percentage"].values[0])
                    y.append(np.mean(experiments_dict[key]["results"]["F-measure corrected"]))
                    e.append(np.std(experiments_dict[key]["results"]["F-measure corrected"]))
                    zero_iterations_data.append(experiments_dict[key]["results"]["F-measure attacked"])
                    if y[-1] < min_corrected_f:
                        min_corrected_f = y[-1]
                    if y[-1] > max_corrected_f:
                        max_corrected_f = y[-1]
            zero_iterations_data = np.concatenate(zero_iterations_data) if len(zero_iterations_data) > 0 else []
            x = [0] + x
            y = [np.mean(zero_iterations_data)] + y
            e = [np.std(zero_iterations_data)] + e
            ax.errorbar(x, y, e, color=colors[color_number], capsize=5,
                        capthick=2 if method == "relabel" else None,
                        linestyle='-.' if method == "remove" else 'solid',
                        label="M: {}, P change: {}".format(method, p_value))
            ax2.errorbar(x, y, e, color=colors[color_number], capsize=5,
                        capthick=2 if method == "relabel" else None,
                        linestyle='-.' if method == "remove" else 'solid')
    plt.xticks([0] + list(visited_values))
    x_points = list(np.linspace(0, max(visited_values), int(max(visited_values)/0.01)))
    y_points = [f_benchmark_mean for i in range(len(x_points))]
    e_points = [f_benchmark_std for i in range(len(x_points))]
    ax.errorbar(x_points, y_points, e_points, linestyle='--', color="blue", capsize=5, label="Original data")
    [ax.errorbar([0], [0], [0.1], color='w', alpha=0, label=' ') for i in range(len(p_values) - 1)]
    ax2.errorbar(x_points, y_points, e_points, linestyle='--', color="blue", capsize=5)
    # ax.plot([0, max(visited_values)], [f_benchmark_mean + f_benchmark_std, f_benchmark_mean + f_benchmark_std], 'g--')
    # ax.plot([0, max(visited_values)], [f_benchmark_mean - f_benchmark_std, f_benchmark_mean - f_benchmark_std], 'r--')
    ax.set_xlim(xmin=-0.01, xmax=max(visited_values) + 0.01)
    ax.set_ylim(ymin=0, ymax=1)
    ax.set(xlabel='Ratio of samples visited at first iteration', ylabel='F-measure')
    ax2.set_xlim(xmin=-0.01, xmax=max(visited_values) + 0.01)
    ax2.set_ylim(ymin=min_corrected_f-0.01, ymax=max_corrected_f + 0.01)
    ax2.set(ylabel="F-measure (zoomed)")
    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(loc='upper left', ncol=len(methods) + 1, prop=fontP)
    f.suptitle("Classification F-measure on test data\n(for data with {}% of class 0 samples)".
              format(100*experiments_dict[keys_of_interest[0]]["parameters"]["ratio"].values[0]))
    f.subplots_adjust(hspace=0)
    return f


def plot_r_for_ratio(experiments_dict, keys_of_interest):
    f, [ax2, ax] = plt.subplots(2, 1, sharex=True)
    f.set_size_inches(7, 8.75)
    r_benchmark = np.concatenate([experiments_dict[key]["results"]["Bayesian risk original"].values for key in keys_of_interest])
    r_benchmark_mean = np.mean(r_benchmark)
    r_benchmark_std = np.std(r_benchmark)
    p_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["p"].values for key in keys_of_interest]))
    visited_values = np.unique(np.concatenate([experiments_dict[key]["parameters"]["visited_percentage"].values for key in keys_of_interest]))
    methods = np.unique(np.concatenate([experiments_dict[key]["parameters"]["method"].values for key in keys_of_interest]))
    min_corrected_r = 1.0
    max_corrected_r = 0.0
    for method in methods:
        for color_number, p_value in enumerate(p_values):
            x = []
            y = []
            e = []
            zero_iterations_data = []
            for key in keys_of_interest:
                if experiments_dict[key]["parameters"]["p"].values[0] == p_value and \
                        experiments_dict[key]["parameters"]["method"].values[0] == method:
                    x.append(experiments_dict[key]["parameters"]["visited_percentage"].values[0])
                    y.append(np.mean(experiments_dict[key]["results"]["Risk corrected"]))
                    e.append(np.std(experiments_dict[key]["results"]["Risk corrected"]))
                    zero_iterations_data.append(experiments_dict[key]["results"]["Risk attacked"])
                    if y[-1] < min_corrected_r:
                        min_corrected_r = y[-1]
                    if y[-1] > max_corrected_r:
                        max_corrected_r = y[-1]
            zero_iterations_data = np.concatenate(zero_iterations_data) if len(zero_iterations_data) > 0 else []
            x = [0] + x
            y = [np.mean(zero_iterations_data)] + y
            e = [np.std(zero_iterations_data)] + e
            ax.errorbar(x, y, e, color=colors[color_number], capsize=5,
                        capthick=2 if method == "relabel" else None,
                        linestyle='-.' if method == "remove" else 'solid',
                        label="M: {}, P change: {}".format(method, p_value))
            ax2.errorbar(x, y, e, color=colors[color_number], capsize=5,
                        capthick=2 if method == "relabel" else None,
                        linestyle='-.' if method == "remove" else 'solid')
    plt.xticks([0] + list(visited_values))
    x_points = list(np.linspace(0, max(visited_values), int(max(visited_values)/0.01)))
    y_points = [r_benchmark_mean for i in range(len(x_points))]
    e_points = [r_benchmark_std for i in range(len(x_points))]
    ax.errorbar(x_points, y_points, e_points, linestyle='--', color="blue", capsize=5, label="Original data")
    [ax.errorbar([0], [0], [0.1], color='w', alpha=0, label=' ') for i in range(len(p_values) - 1)]
    ax2.errorbar(x_points, y_points, e_points, linestyle='--', color="blue", capsize=5)
    # ax.plot([0, max(visited_values)], [r_benchmark_mean + r_benchmark_std, r_benchmark_mean + r_benchmark_std], 'g--')
    # ax.plot([0, max(visited_values)], [r_benchmark_mean - r_benchmark_std, r_benchmark_mean - r_benchmark_std], 'r--')
    ax.set_xlim(xmin=-0.01, xmax=max(visited_values) + 0.01)
    ax.set_ylim(ymin=0, ymax=1)
    ax.set(xlabel='Ratio of samples visited at first iteration', ylabel='Risk')
    ax2.set_xlim(xmin=-0.01, xmax=max(visited_values) + 0.01)
    ax2.set_ylim(ymin=min_corrected_r-0.01, ymax=max_corrected_r + 0.01)
    ax2.set(ylabel="Risk (zoomed)")
    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(loc='upper left', ncol=len(methods) + 1, prop=fontP)
    f.suptitle("Classification Risk on test data\n(for data with {}% of class 0 samples)".
              format(100*experiments_dict[keys_of_interest[0]]["parameters"]["ratio"].values[0]))
    f.subplots_adjust(hspace=0)
    return f


# def get_keys_of_interest(my_dict, parameter_of_interest):
#     # Graph F-measures. One graph per unbalance ratio
#     values = []
#     for element in my_dict.keys():
#         ratio = my_dict[element]["parameters"][para].values[0]
#         if not ratio in ratios:
#             ratios.append(ratio)
#
#     ratios_data = []
#     for ratio in ratios:
#         ratio_data = []
#         for experiment in experiments_dict.keys():
#             if experiments_dict[experiment]["parameters"]["ratio"].values[0] == ratio:
#                 ratio_data.append(experiment)
#         ratios_data.append(ratio_data)

def plot_experiments_summary(experiments_dict, plots_folder=""):
    # Graph F-measures. One graph per unbalance ratio
    ratios = np.unique(np.concatenate([experiments_dict[key]["parameters"]["ratio"].values for key in experiments_dict.keys()]))

    for ratio in ratios:
        ratio_data = []
        for experiment in experiments_dict.keys():
            if experiments_dict[experiment]["parameters"]["ratio"].values[0] == ratio:
                ratio_data.append(experiment)
        f1 = plot_f_for_ratio(experiments_dict, ratio_data)
        f1.savefig(os.path.join(plots_folder, "ratio_{}_classification.png".format(ratio)))
        f1.clear()
        f2 = plot_correction_f_for_ratio(experiments_dict, ratio_data)
        f2.savefig(os.path.join(plots_folder, "ratio_{}_correction.png".format(ratio)))
        f2.clear()
        f3 = plot_r_for_ratio(experiments_dict, ratio_data)
        f3.savefig(os.path.join(plots_folder, "ratio_{}_risk.png".format(ratio)))
        f3.clear()
        plt.close("all")


def compare_gt(original_path, path_to_compare, bg_color="white"):

    destination_path = os.path.split(original_path)[0]
    destination_folder = "%s-VS-%s" % (os.path.split(original_path)[1], os.path.split(path_to_compare)[1])

    image_names = sorted([f for f in os.listdir(original_path)
                          if not f.startswith(".") and f.endswith("_gt.png")],
                         key=lambda f: f.lower())
    if not os.path.exists(os.path.join(destination_path, destination_folder)):
        os.makedirs(os.path.join(destination_path, destination_folder))
    for name in image_names:
        or_gt = cv2.imread(os.path.join(original_path, name))
        ex_gt = cv2.imread(os.path.join(path_to_compare, name))

        if bg_color == "white":
            or_gt = 255 - or_gt
            ex_gt = 255 - ex_gt
            comparative_mask = compare_masks(or_gt, ex_gt, bg_color)
            cv2.imwrite(os.path.join(destination_path, destination_folder, name), comparative_mask)
        else:
            raise ValueError("Values other tan 'white' are not accepted yet.")


def compare_gt_stats(original_path, path_to_compare):

    destination_path = os.path.split(original_path)[0]
    destination_name = "%s-VS-%s.csv" % (os.path.split(original_path)[1], os.path.split(path_to_compare)[1])

    image_names = sorted([f for f in os.listdir(original_path)
                          if not f.startswith(".") and f.endswith("_gt.png")],
                         key=lambda f: f.lower())

    with open(os.path.join(destination_path, destination_name), "w+") as f:
        string_list = ["Image,TP,FP,TN,FN,Total pixels"]
        for name in image_names:
            or_gt = cv2.imread(os.path.join(original_path, name))
            ex_gt = cv2.imread(os.path.join(path_to_compare, name))

            comparative_mask = (compare_masks(or_gt, ex_gt, "black") / 255)
            n_pix = comparative_mask.shape[0] * comparative_mask.shape[1]
            tp = int((comparative_mask[..., 1]).sum())
            fp = int((comparative_mask[..., 2]).sum())
            fn = int((comparative_mask[..., 0]).sum())
            tn = len(np.where(cv2.cvtColor(comparative_mask, cv2.COLOR_BGR2GRAY) == 0)[0])

            string_list.append("%s,%s,%s,%s,%s,%s" % (name, tp, fp, tn, fn, n_pix))
        f.write("\n".join(string_list))

compare_gt_stats("/media/shared_storage/datasets/syncrack_dataset_v3", "/media/shared_storage/datasets/syncrack_dataset_v3")