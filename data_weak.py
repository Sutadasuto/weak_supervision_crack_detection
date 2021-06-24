import  cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow.keras.backend as K

from io import StringIO
from matplotlib.font_manager import FontProperties

from data import compare_masks


def plot2d(x, y, fig, subplot):
    ax = fig.add_subplot(subplot)
    for i in range(len(y)):
        plt.plot(x[i][0], x[i][1], 'o', color="black" if y[i] == 0 else "red", ms=0.5)
    min_x, max_x = np.min(x[:, 0]), np.max(x[:, 0])
    plt.xlim(min_x * 1.01, max_x * 1.01)
    plt.axis('equal')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')


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
    if len(class_0_pixels[1]) > 0:
        plot[class_0_pixels[0], class_0_pixels[1]] = np.concatenate(
            [[[1] for i in range(len(class_0_pixels[0]))],
             1 - canvas_0[class_0_pixels[0], class_0_pixels[1]][..., None],
             1 - canvas_0[class_0_pixels[0], class_0_pixels[1]][..., None]],
            axis=-1
        )
    if len(class_1_pixels[1]) > 0:
        plot[class_1_pixels[0], class_1_pixels[1]] = np.concatenate(
            [1 - canvas_1[class_1_pixels[0], class_1_pixels[1]][..., None],
             1 - canvas_1[class_1_pixels[0], class_1_pixels[1]][..., None],
             [[1] for i in range(len(class_1_pixels[0]))]],
            axis=-1
        )

    return (255*plot).astype(np.uint8)


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
