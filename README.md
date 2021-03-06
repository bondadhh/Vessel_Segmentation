# Implementation of deep learning framework -- Unet, using Keras

## Overview

### Data

The Original data is from VAMPIRE https://vampire.computing.dundee.ac.uk/vesselseg.html. 
They have Ultra Wide field view of Retinal Images which gives much more scope for examining the vessels. Whereas the original Retinal Fundus Images don't have that much large view for examining the vessels. 

### Data augmentation

The data for training contains 8 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

See dataPrepare.ipynb and data.py for detail.


### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 89.6%

Loss function for the training is basically just a binary crossentropy.

## Pre-Processing steps

I have performed Image Sharpening to further enhance the quality of the images. Because the original images are of low quality. 

---

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
