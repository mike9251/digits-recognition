import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.utils import np_utils
from keras.datasets import mnist

import tensorflow as tf
tf.python.control_flow_ops = tf

import cv2

WEIGHTS_FILENAME = './model/weights.h5'

def ConvNet():

    img_rows, img_cols = 28, 28
    
    model = Sequential()

    # Convolution2D(number_filters, row_size, column_size, input_shape=(number_channels, img_row, img_col))

    model.add(Convolution2D(6, 5, 5, input_shape=(1, img_rows, img_cols), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(120, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics = ["accuracy"], optimizer='adadelta')

    return model


def train():
    path="C:/Users/Mike/Documents/Python Scripts/digits-recognition/dataset/mnist.pkl.gz"
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = ConvNet()

    model.fit(X_train, Y_train,
              batch_size=128, 
              nb_epoch=5, 
              verbose=True,
              validation_split=0.1)
    score = model.evaluate(X_test, Y_test, verbose=False)
    print ('Test loss value:', score[0])
    print ('Test accuracy:', score[1])
    model.save_weights(WEIGHTS_FILENAME)

def predict(X):
    batch_size=128
    model = get_model()
    y_pred = model.predict(X, batch_size)
    return y_pred


def get_model():
    model = ConvNet()
    model.load_weights(WEIGHTS_FILENAME)
    return model

#train()