import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import cv2
import numpy as np
import os

from distutils.util import strtobool
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

from callbacks_and_losses import custom_losses
import data

from data_weak import plot_naive
from data import get_image, get_gt_image

from models.available_models import get_models_dict

# Used for memory error in RTX2070
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

models_dict = get_models_dict()


def main(args):
    input_size = (None, None)
    # Load model from JSON file if file path was provided...
    if os.path.exists(args.model):
        try:
            with open(args.model, 'r') as f:
                json = f.read()
            model = model_from_json(json)
            args.model = os.path.splitext(os.path.split(args.model)[-1])[0]
        except JSONDecodeError:
            raise ValueError(
                "JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
                "performed properly." % args.model)
    # ...Otherwise, create model from this project by using a proper key name
    else:
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

    # Load trained weights
    model.load_weights(args.pretrained_weights)

    # decoder = Model([vgg19.input, d_i], output_activation, name="decoder")

    decoder = Model([model.get_layer("encoder").input, model.get_layer("decoder").get_layer("decoder_input").output],
                    model.get_layer("decoder").get_layer("prob_maps").output,
                    name="decoder")
    decoder_output = decoder([model.get_layer("encoder").input,
                              model.get_layer("encoder")(model.get_layer("encoder").input)])

    model = Model(model.get_layer("encoder").input, decoder_output, name=model.name)

    # Model is compiled to provide the desired metrics
    model.compile(optimizer=Adam(lr=1e-4), loss=custom_losses.bce_dsc_loss(3.0),
                  metrics=[custom_losses.dice_coef, keras_metrics.Precision(), keras_metrics.Recall()])

    save_to = "results_%s_%s-nn" % ("-".join(args.dataset_names), args.k)

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    i = 0
    n_images = str(paths.shape[-1])
    for path in np.transpose(paths):
        i += 1
        im_path = path[0]
        gt_path = path[1]
        name = os.path.split(path[1])[-1]
        name, extension = os.path.splitext(name)
        extension = ".png"
        prediction = model.predict(rgb_preprocessor(get_image(im_path))[None, ...])[0, ...]

        pixels = np.reshape(prediction, (prediction.shape[0] * prediction.shape[1], prediction.shape[2]))
        labels = get_gt_image(gt_path).ravel()

        classifier = KNeighborsClassifier(n_neighbors=args.k, n_jobs=-1)
        classifier.fit(pixels, labels)

        print(("\rProcessing image {:0%sd}/{}." % len(n_images)).format(i, n_images), end='')
        new_labels = classifier.predict(pixels)
        new_labels = np.reshape(new_labels, (prediction.shape[0], prediction.shape[1]))
        cv2.imwrite(os.path.join(save_to, name + extension), 255 * new_labels)

    result_string = ""
    for attribute in args.__dict__.keys():
        result_string += "--%s: %s\n" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(save_to, "results.txt"), "w+") as f:
        f.write(result_string.strip())
    print("\rDone.")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', "
                             "'cracktree200', 'text'")
    parser.add_argument("-p", "--dataset_paths", type=str, nargs="+",
                        help="Path to the folders or files containing the respective datasets as downloaded from the "
                             "original source.")
    parser.add_argument("-k", "--k", type=int, default=5,
                        help="K for the K Nearest Neighbors algorithm.")
    parser.add_argument("-w", "--pretrained_weights", type=str,
                        help="Load trained weights from this location.")
    parser.add_argument("-m", "--model", type=str, default="uvgg19", help="Network to use.")

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
