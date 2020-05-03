"""
| *@created on:* 2020-05-04,
| *@author:* shubham,
|
| *Description:* contains the network for the VGG convs
"""

import tensorflow as tf
import numpy as np

net_data = np.load(open("/Users/subhamsingh/Desktop/learning_deep_features_for_discriminative_loacalization/"
                        "alexNet_weights/bvlc_alexnet.npy", "rb"), encoding="latin1", allow_pickle=True).item()

print('net: ', net_data['conv1'])
# layers in weight data = ['fc6', 'fc7', 'fc8', 'conv3', 'conv2', 'conv1', 'conv5', 'conv4']
# we will not be using fc6, fc7, fc8 layer
