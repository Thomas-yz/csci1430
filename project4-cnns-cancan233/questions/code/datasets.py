import os
import gzip
import numpy as np
from skimage import io
from skimage.transform import resize


def load_data_scene(search_path, categories, size, hp):
    images = np.zeros((size * hp.scene_class_count, hp.img_size * hp.img_size))
    labels = np.zeros((size * hp.scene_class_count,), dtype = np.int8)
    for label_no in range(hp.scene_class_count):
        img_path = search_path + categories[label_no]
        img_names = [f for f in os.listdir(img_path) if ".jpg" in f]
        for i in range(size):
            im = io.imread(img_path + "/" + img_names[i])
            im_vector = resize(im, (hp.img_size, hp.img_size)).reshape(1, hp.img_size * hp.img_size)
            index = size * label_no + i
            images[index, :] = im_vector
            labels[index] = label_no
    return images, labels


def get_categories_scene(search_path):
    dir_list = []
    for filename in os.listdir(search_path):
        if os.path.isdir(os.path.join(search_path, filename)):
            dir_list.append(filename)
    return dir_list


def format_data_scene_rec(data_filepath, hp):
    train_path = os.path.join(data_filepath, "train/")
    test_path = os.path.join(data_filepath, "test/")
    categories = get_categories_scene(train_path)
    train_images, train_labels = load_data_scene(train_path, categories, hp.num_train_per_category, hp)
    test_images, test_labels = load_data_scene(test_path, categories, hp.num_test_per_category, hp)
    return train_images, train_labels, test_images, test_labels


def format_data_mnist(data_filepath):
    # Reading in MNIST data.
    # Stolen from CS 1420

    with open(os.path.join(data_filepath, "train-images-idx3-ubyte.gz"), 'rb') as f1,\
            open(os.path.join(data_filepath, "train-labels-idx1-ubyte.gz"), 'rb') as f2:

        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
        train_images = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)
        train_images = np.where(train_images > 99, 1, 0)
        train_labels = np.frombuffer(buf2, dtype='uint8', offset=8)
    with open(os.path.join(data_filepath, "t10k-images-idx3-ubyte.gz"), 'rb') as f1,\
            open(os.path.join(data_filepath, "t10k-labels-idx1-ubyte.gz"), 'rb') as f2:

        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        test_images = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        test_images = np.where(test_images > 99, 1, 0)
        test_labels = np.frombuffer(buf2, dtype='uint8', offset=8)

    return train_images, train_labels, test_images, test_labels
