from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.layers import *
import tensorflow as tf
import keras.backend as K
import numpy as np


def FCN(input_shape=None, weight_decay=0., batch_momentum=0.9, classes=1):
    img_input = Input(shape=input_shape)
    image_size = input_shape[:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
            name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
            padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
            name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
            name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(256, (7, 7), activation='relu', padding='same',
            name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='same',
            name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)

    # classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear',
               padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = UpSampling2D(size=(4, 4))(x)
    model = Model(img_input, x)

    return model
