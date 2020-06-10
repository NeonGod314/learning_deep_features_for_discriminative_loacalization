"""
| *@created on:* 2020-05-04,
| *@author:* shubham,
|
| *Description:* contains the network for the VGG convs
"""

import tensorflow as tf
import tensorboard as tb

from source.data_loader import data_preprocessing, load_cifar10_data, limit_classes_in_data
from source.network import network_1, discriminative_localization_network


"""
Set hyper params
"""

epochs = 100
batch_size = 64
validation_split = 0.05
data_row_limit = 10000
learning_rate = 0.003
class_limit = 3

x_train, y_train, x_test, y_test = load_cifar10_data(limit=data_row_limit)
x_train = data_preprocessing(x_train)
x_train, y_train = limit_classes_in_data(labels=y_train, data=x_train, class_limit=class_limit)
x_test, y_test = limit_classes_in_data(labels=y_test, data=x_test, class_limit=class_limit)


x_test = data_preprocessing(x_test)
y_train = tf.keras.backend.squeeze(y_train, axis=1)
y_test = tf.keras.backend.squeeze(y_test, axis=1)

y_train = tf.one_hot(y_train, depth=class_limit)
y_test = tf.one_hot(y_test, depth=class_limit)

print("label shape: ", y_train.shape)
print("inp data shape: ", x_train.shape)

## train_network 1
log_dir1 = './logs1'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir1, histogram_freq=1)

model_1 = network_1(shape=[32, 32, 3], n_classes=class_limit)

model_1.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.categorical_accuracy])

history = model_1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                      callbacks=[tensorboard_callback])

test_scores = model_1.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

## train network 2
log_dir2 = './logs2'
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)

model_2 = discriminative_localization_network(shape=[32, 32, 3], n_classes=class_limit, network_1=model_1)
model_2.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.categorical_accuracy])
history = model_2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                      callbacks=[tensorboard_callback_1])

test_scores = model_2.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])




