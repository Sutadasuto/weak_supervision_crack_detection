import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from distutils.util import strtobool
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

from callbacks_and_losses import custom_losses
import data

from callbacks_and_losses.custom_calllbacks import EarlyStoppingAtMinValLoss, ReduceLROnPlateau, TensorBoard
from models.available_models import get_models_dict

# Used for memory error in RTX2070
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

models_dict = get_models_dict()


def main(args):
    start = datetime.now().strftime("%d-%m-%Y_%H.%M")
    results_dir = "results_train_bagging_models_%s" % start
    results_dir = "results_train_bagging_models"
    input_size = (None, None)

    model = models_dict[args.model]((input_size[0], input_size[1], 1))
    try:
        # Model name should match with the name of a model from
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/
        # This assumes you used a model with RGB inputs as the first part of your model,
        # therefore your input data should be preprocessed with the corresponding
        # 'preprocess_input' function
        m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
        rgb_preprocessor = getattr(m, "preprocess_input")
    except ModuleNotFoundError:
        rgb_preprocessor = None

    # We don't resize images for training, but provide image patches of reduced size for memory saving
    # An image is turned into this size patches in a chess-board-like approach
    input_size = args.training_crop_size

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    # Data is split into 80% for training data and 20% for validation data. A custom seed is used for reproducibility.
    n_training_images = int(0.2 * paths.shape[1]) if args.dataset_names == ["syncrack"] else int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    whole_training_paths = paths[:, :n_training_images]
    validation_paths = paths[:, n_training_images:]

    # If asked by the user, save the paths of the validation split (useful to validate later)
    if args.save_validation_paths:
        with open("validation_paths.txt", "w+") as file:
            file.write("\n".join([";".join(paths) for paths in validation_paths.transpose()]))
            print("Validation paths saved to 'validation_paths.txt'")

    # If asked by the user, save the paths of the validation split (useful to repeat later)
    if args.save_training_paths:
        with open("training_paths.txt", "w+") as file:
            file.write("\n".join([";".join(paths) for paths in whole_training_paths.transpose()]))
            print("Training paths saved to 'validation_paths.txt'")

    n_splits = args.k
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(whole_training_paths[0, :])

    beggining = 0
    iteration = 0
    for train_index, validation_index in kf.split(whole_training_paths[0, :]):
        tf.keras.backend.clear_session()
        iteration += 1
        if iteration < beggining:
            continue
        print("Iteration %s/%s" % (iteration, n_splits))
        training_paths = whole_training_paths[:, train_index]
        validation_paths = whole_training_paths[:, validation_index]

        results_train_dir = os.path.join(results_dir, str(iteration), "results_training")
        results_train_min_loss_dir = results_train_dir + "_min_val_loss"
        results_validation_dir = os.path.join(results_dir, str(iteration), "results_validation")
        results_validation_min_loss_dir = results_validation_dir + "_min_val_loss"

        # As input images can be of different sizes, here we calculate the total number of patches used for training.
        print("Calculating the total number of samples after cropping and data augmentatiton. "
              "This may take a while, don't worry.")
        n_train_samples = next(data.train_image_generator(training_paths, input_size, args.batch_size,
                                                          count_samples_mode=True, rgb_preprocessor=None,
                                                          data_augmentation=args.use_da))

        while True:
            tf.keras.backend.clear_session()
            print("\nCreating and compiling model.")
            model = models_dict[args.model]((None, None, 3))
            # Model is compiled so it can be trained
            model.compile(optimizer=Adam(lr=args.learning_rate), loss=custom_losses.bce_dsc_loss(args.alpha),
                          metrics=[custom_losses.dice_coef, 'binary_crossentropy',
                                   keras_metrics.Precision(), keras_metrics.Recall()])

            print("\nProceeding to train.")
            # A customized early stopping callback. At each epoch end, the callback will test the current weights on the
            # validation set (using whole images instead of patches) and stop the training if the minimum validation loss hasn't
            # improved over the last 'patience' epochs.
            es = EarlyStoppingAtMinValLoss(validation_paths, file_path='%s_best.hdf5' % iteration, patience=args.patience,
                                           rgb_preprocessor=rgb_preprocessor)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

            # Training begins. Note that the train image generator can use or not data augmentation through the parsed argument
            # 'use_da'
            print("Start!")
            try:
                history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size,
                                                                 rgb_preprocessor=rgb_preprocessor,
                                                                 data_augmentation=args.use_da),
                                    epochs=args.epochs,
                                    verbose=1, callbacks=[es, reduce_lr],
                                    steps_per_epoch=n_train_samples // args.batch_size)

            except KeyboardInterrupt:
                # This assumes a user keyboard interruption, as a hard early stop
                print("Training interrupted. Only the best model will be saved.")
                # Load the weights from the epoch with the minimum validation loss
                model.load_weights('%s_best.hdf5' % iteration)
                args.epochs = 0

            if not es.bad_ending:
                print("Finished!")
                break
            else:
                print("Failed convergence. A new model will be created and trained.")
                del model

        # Save the weights of the last training epoch. If trained on a single epoch, these weights are equal to the
        # best weights (to avoid redundancy, no new weights file is saved)
        if args.epochs > 0:
            model.save_weights("%s.hdf5" % iteration)
            print("Last epoch's weights saved.")

        print("Evaluating the model...")
        print("On training paths:")
        data.save_results_on_paths(model, training_paths, results_train_dir)
        if args.epochs > 0:
            os.replace("%s.hdf5" % iteration, os.path.join(results_train_dir, "%s.hdf5" % iteration))
        else:
            model.save_weights(os.path.join(results_train_dir, "%s.hdf5" % iteration))
        print("\nOn validation paths:")
        data.save_results_on_paths(model, validation_paths, results_validation_dir)

        # Evaluate the model quantitatively by using the tensorflow model's metrics
        metrics = model.evaluate(
            x=data.validation_image_generator(validation_paths, batch_size=1, rgb_preprocessor=rgb_preprocessor),
            steps=validation_paths.shape[1])
        result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
        for idx, metric in enumerate(model.metrics_names):
            result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
        for attribute in args.__dict__.keys():
            result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
        with open(os.path.join(results_validation_dir, "results.txt"), "w") as f:
            f.write(result_string.strip())
        data.calculate_thresholded_dsc_from_image_folder(results_validation_dir)

        # If the model was trained more than one epoch, evaluate the best validation weights in the same validation data.
        # Otherwise, the best validation weights are the weights at the end of the only training epoch
        if args.epochs > 1:
            # Load results using the min val loss epoch's weights
            model.load_weights('%s_best.hdf5' % iteration)
            print("Evaluating the model with minimum validation loss...")
            print("On training paths:")
            data.save_results_on_paths(model, training_paths, results_train_min_loss_dir)
            print("\nOn validation paths:")
            data.save_results_on_paths(model, validation_paths, results_validation_min_loss_dir)
            os.replace('%s_best.hdf5' % iteration, os.path.join(results_train_min_loss_dir, '%s_best.hdf5' % iteration))
            metrics = model.evaluate(
                x=data.validation_image_generator(validation_paths, batch_size=1, rgb_preprocessor=rgb_preprocessor),
                steps=validation_paths.shape[1])
            result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
            for idx, metric in enumerate(model.metrics_names):
                result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
            for attribute in args.__dict__.keys():
                result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
            with open(os.path.join(results_validation_min_loss_dir, "results.txt"), "w") as f:
                f.write(result_string.strip())
            data.calculate_thresholded_dsc_from_image_folder(results_validation_min_loss_dir)

            # If trained more than one epoch, save the training history as csv and plot it
            print("\nPlotting training history...")
            import pandas as pd
            pd.DataFrame.from_dict(history.history).to_csv(os.path.join(results_dir, str(iteration), "training_history.csv"), index=False)
            # summarize history for loss
            for key in history.history.keys():
                plt.plot(history.history[key])
            plt.ylim((0.0, 1.0 + args.alpha))
            plt.title('model losses')
            plt.ylabel('value')
            plt.xlabel('epoch')
            plt.legend(history.history.keys(), loc='upper left')
            plt.savefig(os.path.join(results_dir, str(iteration), "training_losses.png"))
            plt.close()

        del model


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', "
                             "'cracktree200', 'text'")
    parser.add_argument("-p", "--dataset_paths", type=str, nargs="+",
                        help="Path to the folders or files containing the respective datasets as downloaded from the "
                             "original source.")
    parser.add_argument("-k", "--k", type=int, default=10, help="Number of folds.")
    parser.add_argument("-m", "--model", type=str, default="uvgg19",
                        help="Network to use. It can be either a name from 'models.available_models.py' or a path to a "
                             "json file.")
    parser.add_argument("-cs", "--training_crop_size", type=int, nargs=2, default=[256, 256],
                        help="For memory efficiency and being able to admit multiple size images,"
                             "subimages are created by cropping original images to this size windows")
    parser.add_argument("-a", "--alpha", type=float, default=3.0,
                        help="Alpha for objective function: BCE_loss + alpha*DICE")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("-e", "--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience.")
    parser.add_argument("-w", "--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")
    parser.add_argument("-da", "--use_da", type=str, default="False", help="If 'True', training will be done using data "
                                                                   "augmentation. If 'False', just raw images will be "
                                                                   "used.")
    parser.add_argument("--save_validation_paths", type=str, default="False", help="If 'True', a text file "
                                                                                   "'validation_paths.txt' containing "
                                                                                   "the paths of the images used "
                                                                                   "for validating will be saved in "
                                                                                   "the project's root.")
    parser.add_argument("--save_training_paths", type=str, default="False", help="If 'True', a text file "
                                                                                   "'training_paths.txt' containing "
                                                                                   "the paths of the images used "
                                                                                   "for training will be saved in "
                                                                                   "the project's root.")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
