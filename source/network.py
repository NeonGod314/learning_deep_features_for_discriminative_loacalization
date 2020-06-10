"""
| *@created on:* 2020-05-04,
| *@author:* shubham,
|
| *Description:* contains the network for the VGG convs
"""

import tensorflow as tf
import numpy as np


# net_data = np.load(open("/Users/subhamsingh/Desktop/"
#                         "alexNet_weights/bvlc_alexnet.npy", "rb"), encoding="latin1", allow_pickle=True).item()

# print('net: ', net_data['conv1'])
# layers in weight data = ['fc6', 'fc7', 'fc8', 'conv3', 'conv2', 'conv1', 'conv5', 'conv4']
# we will not be using fc6, fc7, fc8 layer


def network_1(shape, n_classes):
    """
    network construction
    """
    inp_x = tf.keras.layers.Input(shape=shape, name='input_1')
    conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=6, padding='same', strides=1, name='conv1')(inp_x)
    norm1 = tf.keras.layers.BatchNormalization(name='norm1')(conv1)
    max1 = tf.keras.layers.AveragePooling2D(pool_size=2, padding='valid', name='max1')(norm1)

    conv2 = tf.keras.layers.Conv2D(filters=9, kernel_size=5, padding='same', strides=1, name='conv2')(max1)
    norm2 = tf.keras.layers.BatchNormalization(name='norm2')(conv2)

    conv_int = tf.keras.layers.Conv2D(filters=9, kernel_size=5, padding='same', strides=1, name='conv_int')(norm2)

    conv3 = tf.keras.layers.Conv2D(filters=18, kernel_size=3, padding='same', strides=1, name='conv3')(conv_int)

    reshape = tf.keras.layers.Flatten()(conv3)
    dense_1 = tf.keras.layers.Dense(units=32, use_bias=True, activation='relu')(reshape)
    dense_2 = tf.keras.layers.Dense(units=20, use_bias=True, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(units=15, use_bias=True, activation='relu')(dense_2)

    y = tf.keras.layers.Dense(units=n_classes, use_bias=True, activation='softmax')(dense_3)

    return tf.keras.Model(inputs=inp_x, outputs=y, name='network_1')


def discriminative_localization_network(shape, n_classes, network_1):
    """
    network construction
    """
    inp_x = tf.keras.layers.Input(shape=shape, name='input_1')
    conv1 = network_1.get_layer('conv1')(inp_x)
    norm1 = network_1.get_layer('norm1')(conv1)
    max1 = network_1.get_layer('max1')(norm1)

    conv2 = network_1.get_layer('conv2')(max1)
    norm2 = network_1.get_layer('norm2')(conv2)

    conv_int = network_1.get_layer('conv_int')(norm2)

    conv3 = network_1.get_layer('conv3')(conv_int)

    GAP = tf.keras.layers.GlobalAveragePooling2D()(conv3)

    reshape = tf.keras.layers.Flatten()(GAP)
    y = tf.keras.layers.Dense(units=n_classes, use_bias=True, activation='softmax')(reshape)

    return tf.keras.Model(inputs=inp_x, outputs=y, name='discriminative_localization')
