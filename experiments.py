import os

from data_weak import get_experiments_dict, plot_experiments_summary
from main import run_experiment

# Data parameters
ratios = [0.99, 0.98, 0.97]  # percentage of black dots
ratios = [0.9, 0.99]  # percentage of black dots
n_samples = 1000  # size of the training dataset
n_samples_test = 1000  # size of the test set
ps = [i / 10.0 for i in range(1, 5)]  # independent probability of randomly changing the label of a given sample
ps = [i / 10.0 for i in range(2, 4)]  # independent probability of randomly changing the label of a given sample
visited_percentages = [i / 10.0 for i in range(1, 6)]  # percentage of data points to check for labeling correction
visited_percentages = [i / 10.0 for i in range(2, 4)]  # percentage of data points to check for labeling correction

# Correction algorithm parameters
methods = ["relabel", "remove"]  # either to 'relabel' or 'remove' suspicious samples
# methods = ["relabel"]  # either to 'relabel' or 'remove' suspicious samples
threshold = 0.9  # Algorithm finishes if the ratio F-corrected/F-original on test data is at least 'threshold'
limit = 100  # Algorithm finishes if this number of iterations is reached
patience = 10  # Algorithm finishes if F-corrected on test data hasn't improved during 'patience' consecutive iterations

# Training parameters
epochs = 500
verbose = 0

n_experiments = 3
results_folder = "results"


def extract_info(path):
    with open(path, "r") as results_file:
        results = results_file.read()
    data = results.split("\n")
    names = []
    values = []
    for data_point in data:
        name, value = data_point.split(": ")
        names.append(name)
        values.append(value)
    return names, values


def save_summary(input_txt_file, output_csv_file, experiment, iteration):
    names, values = extract_info(input_txt_file)

    new_string = ""
    if iteration == 0:
        experiment_name = ",".join(["Experiment{:03d}".format(experiment) for i in range(len(names))])
        new_string += "\n%s\n%s" % (experiment_name, ",".join(names))
        if experiment == 1:
            new_string = new_string.strip()
    new_string += "\n%s" % ",".join(values)

    with open(output_csv_file, "a+") as results_summary_file:
        results_summary_file.write(new_string)


def save_parameters(output_csv_file, **parameters_and_names):
    new_string = ""
    if experiment == 1:
        new_string += ",".join(parameters_and_names.keys())
    values = [str(parameters_and_names[key]) for key in parameters_and_names.keys()]
    new_string += "\n%s" % ",".join(values)
    with open(output_csv_file, "a+") as parameters_file:
        parameters_file.write(new_string)


experiment = 1
for method in methods:
    for ratio in ratios:
        for p in ps:
            for visited_percentage in visited_percentages:
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)
                save_parameters(
                    os.path.join(results_folder, "experiment_parameters.csv"),
                    experiment=experiment,
                    method=method,
                    ratio=ratio,
                    p=p,
                    visited_percentage=visited_percentage
                )
                for iteration in range(n_experiments):
                    folder_name = os.path.join(results_folder, "Experiment{:03d}".format(experiment),
                                               "Iteration{:03d}".format(iteration + 1))
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    run_experiment(ratio, n_samples, n_samples_test, p, visited_percentage, method, threshold, limit,
                                   patience, epochs, verbose, results_folder=folder_name)
                    save_summary(
                        os.path.join(folder_name, "results_comparison.txt"),
                        os.path.join(results_folder, "results_summary.csv"),
                        experiment=experiment,
                        iteration=iteration
                    )
                experiment += 1

my_dict = get_experiments_dict(os.path.join(results_folder, "results_summary.csv"),
                               os.path.join(results_folder, "experiment_parameters.csv"))
plot_experiments_summary(my_dict, results_folder)
