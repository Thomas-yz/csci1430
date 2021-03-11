"""
CSCI 1430 Deep Learning project
By Ruizhao Zhu, Aaron Gokaslan, James Tompkin

This executable is used to launch the model on a given dataset. Additionally, it the data processing
parsing data into numpy arrays.

Usage:
    python main.py --dataset [DATA] --mode [MODE] --data [DATA_PATH]
        DATA | "mnist" or "scenerec"
        MODE | "nn" or "svm"
        DATA_PATH | path to dataset
"""
import os
import sys
import argparse

import hyperparameters as hp
from model import Model
from datasets import format_data_scene_rec, format_data_mnist
from datetime import datetime

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="Designates dataset to use. Valid values: mnist, scenerec",
)

parser.add_argument(
    "--mode",
    required=True,
    help="Designates classifier to use. Valid values: nn, nn+svm",
)

parser.add_argument("--data", default="../../data", help="Dataset relative filepath")


ARGS = parser.parse_args()


def main():
    datasets = {"mnist", "scenerec"}
    mode = {"nn", "nn+svm"}

    if ARGS.dataset not in datasets:
        raise ValueError("Data must be one of %r.", datasets)

    if ARGS.mode not in mode:
        raise ValueError("Mode must be one of %r.", mode)

    if ARGS.dataset == "scenerec":
        train_images, train_labels, test_images, test_labels = format_data_scene_rec(
            ARGS.data, hp
        )
        num_classes = hp.scene_class_count
    else:
        train_images, train_labels, test_images, test_labels = format_data_mnist(
            ARGS.data
        )
        num_classes = hp.mnist_class_count

    model = Model(train_images, train_labels, num_classes, hp)

    if ARGS.mode == "nn":
        model.train_nn()
        accuracy = model.accuracy_nn(test_images, test_labels)
        print("nn model training accuracy: {:.0%}".format(accuracy))
    else:
        model.train_nn()
        model.train_svm()
        accuracy = model.accuracy_svm(test_images, test_labels)
        print("nn+svm model training accuracy: {:.0%}".format(accuracy))


if __name__ == "__main__":

    start = datetime.now()
    main()
    print(datetime.now() - start)
