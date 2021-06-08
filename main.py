from datetime import datetime
from math import isnan
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

import tensorflow as tf
import tensorflow.keras.metrics as keras_metrics
import os

from data_weak import *

from gradients_and_influence import grads, get_determination_scores

# Data parameters
ratio = 0.9  # percentage of black dots
n_samples = 1000  # size of the training dataset
n_samples_test = 1000  # size of the test set
p = 0.3  # independent probability of randomly changing the label of a given sample
visited_percentage = 0.06  # percentage of data points to check for labeling correction
mean_0, cov_0 = [-1, 0], [[1, 0], [0, 1]]  # Distribution of class 0 samples
mean_1, cov_1 = [1, 0], [[1, 0], [0, 1]]  # Distribution of class 1 samples

# Correction algorithm parameters
method = "relabel"  # either to 'relabel' or 'remove' suspicious samples
threshold = 0.9  # Algorithm finishes if the ratio F-corrected/F-original on test data is at least 'threshold'
limit = 100  # Algorithm finishes if this number of iterations is reached
patience = 10  # Algorithm finishes if F-corrected on test data hasn't improved during 'patience' consecutive iterations

# Training parameters
epochs = 500
verbose = 0
batch_size = 128

results_folder = None  # Path to the folder in which the results will be saved (if None, it becomes data_time)

"""
Define model
"""
metrics = [keras_metrics.Precision(), keras_metrics.Recall()]


def create_model(x, metrics):
    model = Sequential()
    model.add(Dense(1))  # A single neuron
    model.add(Activation("sigmoid"))  # Bound the output to (0, 1)

    model(x[:1, :])  # Evaluate on asingle data point to determine input size
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
    return model


def run_experiment(ratio, n_samples, n_samples_test, p, visited_percentage, method, threshold, limit, patience, epochs,
                   verbose, results_folder=None):
    """ 
    :param float ratio: Ratio between black (class 0) and red (class 1) points
    :param int n_samples: Size of the training set
    :param int n_samples_test: Size of the test set
    :param float p: Independent probability of randomly changing a sample label
    :param float visited_percentage: Percentage of samples to visit at the first iteration of the correction algorithm
    :param string method: Either to 'remove' or 'relabel' a suspect sample in the correction algorithm
    :param float threshold: Correction algorithm finishes if the rate F-corrected/F-original on test data is at least this value
    :param int limit: Correction algorithm finishes if this number of iterations is reached
    :param int patience: Correction algorithm finishes if this number of iterations is reached
    :param int epochs: Train each model during this number of epochs
    :param int verbose: Training verbosity level
    :param string results_folder: Path to the folder where the results should be saved. If None, it becomes date_time
    """
    """
    Create distributions. Two for training, two for test.
    """
    # Distributions for training
    n_samples_0 = int(round(n_samples * ratio))
    x_0, y_0 = create_distribution(mean_0, cov_0, n_samples_0, label=0)

    n_samples_1 = int(round(n_samples * (1 - ratio)))
    x_1, y_1 = create_distribution(mean_1, cov_1, n_samples_1, label=1)

    x = np.concatenate([x_0, x_1], axis=0)
    y = np.concatenate([y_0, y_1], axis=0)
    del x_0, x_1, y_0, y_1
    flipped = np.zeros(y.shape, dtype=np.int)

    # Plot the distributions in a plane
    # fig = plt.figure()
    # plot2d(x, y, fig, 221)  # Show the distributions before changing labels

    # Distributions for testing
    n_samples_0 = int(round((n_samples_test * ratio)))
    x_0_test, y_0_test = create_distribution(mean_0, cov_0, n_samples_0, label=0)

    n_samples_1 = int(round(n_samples_test * (1 - ratio)))
    x_1_test, y_1_test = create_distribution(mean_1, cov_1, n_samples_1, label=1)

    x_test = np.concatenate([x_0_test, x_1_test], axis=0)
    y_test = np.concatenate([y_0_test, y_1_test], axis=0)
    del x_0_test, x_1_test, y_0_test, y_1_test

    """
    Run on original data: train model; calculate gradients; calculate and plot influences
    """
    model_original = create_model(x, metrics)
    # Save initial weights for later. All models will be trained with these initial weights
    initial_weights = model_original.get_weights()
    print("Training...", end="")
    class_weight = get_class_weights(y)  # As classes are unbalanced, we weight them during training
    history_original = model_original.fit(x, y, class_weight=class_weight, batch_size=batch_size, epochs=epochs,
                                          verbose=verbose)
    print(" done!")

    nb = GaussianNB()
    print("Training Bayesian classifier...", end="")
    # nb.fit(x, y, get_sample_weights(y))
    nb.fit(x, y)
    print(" done!")

    # Calculate first order gradient (Jacobian) and Hessian matrix
    print("Calculating gradients and hessians...", end="")
    g, h = grads(x, y, model_original)
    print(" done!")

    # Create a summary of the relevant data. Each row contains:
    # x_0, x_1, y, y_hat, average influence, label was originally flipped?
    # for a single data point
    data_original = create_summary(x, y, flipped, model_original, g, h)
    # plot3d(data_original, fig, 222)  # Show the influence scores of 'data original' in a 3d chart

    """
    Run on attacked (flipped labels) data: flip data, train model, calculate gradients, calculate and plot influences
    """
    # Flip labels with probability p
    y_flipped, flipped = flip_random_labels(y, p)
    # Clone the input features, so we can compare altered data points to the original dataset
    x_flipped = np.copy(x)
    # plot2d(x_flipped, y_flipped, fig, 223)  # Show the distributions after changing labels

    model_attacked = create_model(x_flipped, metrics)
    model_attacked.set_weights(initial_weights)
    class_weight = get_class_weights(y_flipped)
    print("Training...", end="")
    history_attacked = model_attacked.fit(x_flipped, y_flipped, class_weight=class_weight, batch_size=batch_size,
                                          epochs=epochs, verbose=verbose)
    print(" done!")

    nb_attacked = GaussianNB()
    print("Training Bayesian classifier...", end="")
    nb_attacked.fit(x_flipped, y_flipped, get_sample_weights(y_flipped))
    print(" done!")

    print("Calculating gradients and hessians...", end="")
    g, h = grads(x_flipped, y_flipped, model_attacked)
    print(" done!")

    data_attacked = create_summary(x_flipped, y_flipped, flipped, model_attacked, g, h)
    n_flipped = np.sum(flipped)
    # plot3d(data_attacked, fig, 224)  # Show the influence scores of 'data_attacked' in a 3d chart

    """
    Show plots
    """
    # plt.show()

    """
    Correction process
    """
    print("Evaluation original model:")
    performance_original = model_original.evaluate(x_test, y_test)
    precision_original = performance_original[1]
    recall_original = performance_original[2]
    f_original = 2 * precision_original * recall_original / (precision_original + recall_original)
    if isnan(f_original):
        f_original = 1e-10
    f_original_nb = f1_score(y_test, nb.predict(x_test))
    bayesian_risk = calculate_risk(y_test, nb.predict(x_test))
    bayesian_risk_nn_orginal = calculate_risk(y_test, (model_original.predict(x_test) >= 0.5).astype(np.int8))

    # Define the date and time of start to use them as path name to save the results
    start = datetime.now().strftime("%d-%m-%Y_%H.%M") if results_folder is None else results_folder
    if not os.path.exists(start):
        os.makedirs(start)

    min_x = min(min(x[:, 0]), min(x_test[:, 0]))
    max_x = max(max(x[:, 0]), max(x_test[:, 0]))
    x_range = [min_x, max_x]
    min_y = min(min(x[:, 1]), min(x_test[:, 1]))
    max_y = max(max(x[:, 1]), max(x_test[:, 1]))
    y_range = [min_y, max_y]

    iteration = 0
    best_corrected_f = -1
    patience_counter = 0
    n_changed_samples = 0
    n_flipped_corrected = 0
    fs_corrected = []
    iterations = []
    while True:

        print("Evaluating new model:")
        performance_attacked = model_attacked.evaluate(x_test, y_test)
        precision_corrected = performance_attacked[1]
        recall_corrected = performance_attacked[2]
        f_corrected = 2 * precision_corrected * recall_corrected / (precision_corrected + recall_corrected)
        f_corrected_nb = f1_score(y_test, nb_attacked.predict(x_test))
        bayesian_risk_nn_corrected = calculate_risk(y_test, (model_attacked.predict(x_test) >= 0.5).astype(np.int8))
        bayesian_risk_corrected = calculate_risk(y_test, nb_attacked.predict(x_test))
        if isnan(f_corrected):
            f_corrected = 1e-10
        if iteration == 0:
            f_attacked = f_corrected
            precision_attacked = precision_corrected
            recall_attacked = recall_corrected
            f_attacked_nb = f_corrected_nb
            bayesian_risk_nn_attacked = bayesian_risk_nn_corrected
            bayesian_risk_attacked = bayesian_risk_corrected

            def write_results():
                with open("%s/results_comparison.txt" % start, "w+") as text_file:
                    text_file.write(
                        "Precision original: {:.4f}\n"
                        "Recall original: {:.4f}\n"
                        "F-measure original: {:.4f}\n"
                        "Risk original: {:.4f}\n"
                        "Bayesian risk original: {:.4f}\n"
                        "Precision attacked: {:.4f}\n"
                        "Recall attacked: {:.4f}\n"
                        "F-measure attacked: {:.4f}\n"
                        "Risk attacked: {:.4f}\n"
                        "Bayesian risk attacked: {:.4f}\n"
                        "Precision corrected: {:.4f}\n"
                        "Recall corrected: {:.4f}\n"
                        "F-measure corrected: {:.4f}\n"
                        "Risk corrected: {:.4f}\n"
                        "Bayesian risk corrected: {:.4f}\n"
                        "Iterations: {}\n"
                        "Attacked samples: {}\n"
                        "Corrected samples: {}\n"
                        "True positives: {}\n"
                        "Precision: {:.4f}\n"
                        "Recall: {:.4f}\n"
                        "F-measure: {:.4f}".
                            format(precision_original, recall_original, f_original, bayesian_risk_nn_orginal,
                                   bayesian_risk, precision_attacked, recall_attacked, f_attacked,
                                   bayesian_risk_nn_attacked, bayesian_risk_attacked, precision_corrected,
                                   recall_corrected, f_corrected, bayesian_risk_nn_corrected, bayesian_risk_corrected,
                                   iteration, n_flipped, n_changed_samples, n_flipped_corrected, precision, recall, f))

        if f_corrected > best_corrected_f:
            patience_counter = 0
            best_corrected_f = f_corrected
        else:
            patience_counter += 1
        fs_corrected.append(f_corrected)
        iterations.append(iteration)

        """
        Here we plot only the MLP decision boundary
        """
        # fig2 = plt.figure()
        # fig2.suptitle("Iteration {}".format(iteration), weight="bold", size=14)
        # plot_decision_boundary(x, y, model_original, fig2, 221, title="Training data - original")
        # plot_decision_boundary(x_test, y_test, model_original, fig2, 222,
        #                        title="Test data - F={:.4f}".format(f_original))
        # plot_decision_boundary(x_flipped, y_flipped, model_attacked, fig2, 223, title="Training data - attacked")
        # plot_decision_boundary(x_test, y_test, model_attacked, fig2, 224,
        #                        title="Test data - F={:.4f}".format(f_corrected))
        # plt.tight_layout()
        # fig2.savefig("{}/{:03d}.png".format(start, iteration))
        # fig2.clear()

        """
        Here we plot the MLP decision boundary as well as the NB probability map
        """
        fig2 = plt.figure()
        fig2.set_size_inches(16, 8)
        fig2.suptitle("Iteration {}".format(iteration), weight="bold", size=14)
        plot_decision_boundary(x, y, model_original, fig2, 241, title="Training data MLP - original", x_range=x_range,
                               y_range=y_range)
        plot_decision_boundary(x_test, y_test, model_original, fig2, 242,
                               title="Test data\nF={:.4f}, R={:.4f}".format(f_original, bayesian_risk_nn_orginal),
                               x_range=x_range, y_range=y_range)
        plot_probability_map(x, y, nb, fig2, 243, title="Training data NB - original", x_range=x_range, y_range=y_range)
        plot_probability_map(x_test, y_test, nb, fig2, 244, title="Test data\nF={:.4f}, R={:.4f}".format(f_original_nb,
                                bayesian_risk), x_range=x_range, y_range=y_range)
        plot_decision_boundary(x_flipped, y_flipped, model_attacked, fig2, 245, title="Training data MLP - attacked",
                               x_range=x_range, y_range=y_range)
        plot_decision_boundary(x_test, y_test, model_attacked, fig2, 246,
                               title="Test data\nF={:.4f}, R={:.4f}".format(f_corrected, bayesian_risk_nn_corrected),
                               x_range=x_range, y_range=y_range)
        plot_probability_map(x_flipped, y_flipped, nb_attacked, fig2, 247, title="Training data NB - attacked",
                             x_range=x_range, y_range=y_range)
        plot_probability_map(x_test, y_test, nb_attacked, fig2, 248,
                             title="Test data\nF={:.4f}, R={:.4f}".format(f_corrected_nb, bayesian_risk_corrected),
                             x_range=x_range, y_range=y_range)
        plt.tight_layout()
        fig2.savefig("{}/{:03d}.png".format(start, iteration))
        fig2.clear()

        if (
                f_corrected / f_original >= threshold or iteration >= limit or patience_counter > patience) and iteration > 0:
            print("Finishing condition met!")
            precision = n_flipped_corrected / n_changed_samples
            recall = n_flipped_corrected / n_flipped
            f = 2 * precision * recall / (precision + recall)
            if isnan(f):
                f = 1e-10
            write_results()
            break

        iteration += 1
        print("Iteration {}...".format(iteration))
        determination_scores = get_determination_scores(data_attacked)
        new_order = (-determination_scores).argsort()

        # points_to_visit = int(visited_percentage * len(y_flipped))
        points_to_visit = int(visited_percentage ** iteration * len(y_flipped))
        labels = data_attacked[new_order[:points_to_visit], 2]
        predictions = (data_attacked[new_order[:points_to_visit], 3] > 0.5).astype(np.float)

        suspects = new_order[np.where(labels != predictions)]
        n_changed_samples += len(suspects)
        n_flipped_corrected += np.sum(data_attacked[suspects, 5].astype(np.int))
        if len(suspects) == 0:
            print("No more suspects to correct!")
            precision = n_flipped_corrected / n_changed_samples
            recall = n_flipped_corrected / n_flipped
            f = 2 * precision * recall / (precision + recall)
            if isnan(f):
                f = 1e-10
            write_results()
            break

        if method == "relabel":
            y_flipped[suspects] = 1 - y_flipped[suspects]
        elif method == "remove":
            y_flipped = np.delete(y_flipped, suspects, 0)
            x_flipped = np.delete(x_flipped, suspects, 0)
            flipped = np.delete(flipped, suspects, 0)

        model_attacked = create_model(x_flipped, metrics)
        model_attacked.set_weights(initial_weights)
        print("Training with corrected labels... ", end="")
        class_weight = get_class_weights(y_flipped)
        history_attacked = model_attacked.fit(x_flipped, y_flipped, class_weight=class_weight, batch_size=batch_size,
                                              epochs=epochs, verbose=verbose)
        print("done!")

        nb_attacked = GaussianNB()
        print("Training Bayesian classifier with corrected labels...", end="")
        nb_attacked.fit(x_flipped, y_flipped, get_sample_weights(y_flipped))
        print(" done!")

        print("Calculating gradients and hessians...", end="")
        g, h = grads(x_flipped, y_flipped, model_attacked)
        print(" done!")
        data_attacked = create_summary(x_flipped, y_flipped, flipped, model_attacked, g, h)
    # Clean Tensorflow session to avid memory issues
    tf.keras.backend.clear_session()

    # Graph the evolution of the F-measure on the test data training with corrected data
    f, axarr = plt.subplots(2, 1, sharex=True)
    f.suptitle('Test F-measure along correction iterations')
    axarr[1].plot(iterations, fs_corrected)
    axarr[1].plot([iterations[0], iterations[-1]], [f_original, f_original], 'r--')
    axarr[1].set(xlabel='Iterations', ylabel='F-measure')
    axarr[1].grid()
    axarr[1].set_ylim(0, 1.0)
    axarr[1].set_xlim(xmin=0)
    plt.xticks(iterations)
    plt.legend(["Corrected data", "Original data"], loc='upper left')
    axarr[0].plot(iterations, fs_corrected)
    axarr[0].plot([iterations[0], iterations[-1]], [f_original, f_original], 'r--')
    axarr[0].set(ylabel='F-measure (zoom)')
    axarr[0].grid()
    axarr[0].set_ylim(min(fs_corrected) - 0.01, max(max(fs_corrected), f_original) + 0.01)
    axarr[0].set_xlim(xmin=0)
    f.subplots_adjust(hspace=0)
    f.savefig("{}/{}.png".format(start, "F-measure_evolution"))
    f.clear()

    # Close figures
    plt.close("all")


if __name__ == "__main__":
    run_experiment(ratio, n_samples, n_samples_test, p, visited_percentage, method, threshold, limit, patience, epochs,
                   verbose, results_folder=None)
