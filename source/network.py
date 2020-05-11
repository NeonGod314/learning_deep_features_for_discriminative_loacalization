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


def load_cifar10_data(limit=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(y_train[1:10])
    if limit is None:
        return (x_train, y_train, x_test, y_test)
    else:
        return (x_train[:limit], y_train[:limit], x_test[:limit], y_test[:limit])


def network(shape, n_classes):
    inp_x = tf.keras.layers.Input(shape=shape)
    conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=6, padding='same', strides=1)(inp_x)
    max1 = tf.keras.layers.AveragePooling2D(pool_size=2, padding='valid')(conv1)

    conv2 = tf.keras.layers.Conv2D(filters=9, kernel_size=5, padding='same', strides=1)(max1)

    # GAP
    conv3 = tf.keras.layers.Conv2D(filters=18, kernel_size=3, padding='same', strides=1)(conv2)

    GAP = tf.keras.layers.GlobalAveragePooling2D()(conv3)

    reshape = tf.keras.layers.Flatten()(GAP)
    y = tf.keras.layers.Dense(units=n_classes, use_bias=True, activation='softmax')(reshape)

    return tf.keras.Model(inputs=inp_x, outputs=y, name='discriminative_localization')


def data_preprocessing(img_data):
    return img_data / 255.


if __name__ == '__main__':
    ## hyper params
    epochs = 500
    batch_size = 32
    validation_split = 0.1
    data_row_limit = 1000
    learning_rate = 0.01

    x_train, y_train, x_test, y_test = load_cifar10_data(limit=data_row_limit)
    x_train = data_preprocessing(x_train)
    x_test = data_preprocessing(x_test)
    y_train = tf.keras.backend.squeeze(y_train, axis=1)
    y_test = tf.keras.backend.squeeze(y_test, axis=1)

    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    # y_train = tf.reshape(y_train, shape=[1000, -1])
    # y_test = tf.reshape(y_test, shape=[1000, -1])

    print("label shape: ", y_train.shape)
    print("inp data shape: ", x_train.shape)

    model = network(shape=[32, 32, 3], n_classes=10)
    print(tf.keras.utils.plot_model(model, 'disc_local.png', show_shapes=True))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    test_scores = model.evaluate(x_test, y_test)
    train_scores = model.evaluate(x_train, y_train)

    print('train loss accuracy:', train_scores)
    print('Test loss accuracy:', test_scores)
