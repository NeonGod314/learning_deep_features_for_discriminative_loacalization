"""
| *@created on:* 2020-05-04,
| *@author:* shubham,
|
| *Description:* contains data_loader modules
"""

import tensorflow as tf


def load_cifar10_data(limit = None):
    """
    loads data
    :param limit: limit how much data you want
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(y_train[1:10])
    if limit is None:
        return (x_train, y_train, x_test, y_test)
    else:
        return (x_train[:limit], y_train[:limit], x_test[:limit], y_test[:limit])


def data_preprocessing(img_data):
    """
    img data preprocessing
    :param img_data: image data
    """
    return img_data/255.


def limit_classes_in_data(labels, data, class_limit):
    indexes = tf.where(labels < class_limit)[:, 0]
    new_labels = labels[indexes, :]
    new_data = data[indexes, :]

    return new_data, new_labels

