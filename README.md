## Tensorflow implementation of Learning Deep Features for Discriminative Localization"

Model will be trained on ILSVRC 2014 dataset

### Paper review/ introduction

The paper introduces a way to implement object localization without using annotated
data or data with bounding boxes and is based on the idea that units become more and more
discriminative as the depth of the n/w increases.

For the example implementation, I will be using ResNet50 net after clipping its FC layers.

#### Related concepts:
##### 1.  Class Activation Mapping
A class activation map for a particular category indicates the discriminative image regions
used by the CNN to identify the category. 
##### 2. Global Average Pooling
In here in this case, we take the average of all the elements of a unit of last conv output.

Note: Why GAP and not GMP?  <br>
GAP loss encourages the network to identify the extent of the object whereas GMP encourages
the network to identify just one discriminative part.

Thick architecture: 
![basic_arch](imgs/basic_arch.png)
![basic_arch](https://github.com/NeonGod314/learning_deep_features_for_discriminative_loacalization/blob/master/imgs/basic_arch.png?raw=true)


#### Implementation Notes:
AlexNet is not available in tf. 
Download it from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
From the above download, we will be using weights for only ['conv3', 'conv2', 'conv1', 'conv5', 'conv4']

## Update:
(2:31pm, May12, 20) : <BR />1. Basic implementation tested on Colab with K80 gpu.
    For 5000 data row, 2000 epochs, 64 batch size - acc: 0.40, val_acc: 0.36. 
    "Seems to be overfitting for less amount of data for 10 classes"
    <BR />1. Increasing the data row to 15000 seems to reduce overfitting significantly